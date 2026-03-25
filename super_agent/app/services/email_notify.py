from __future__ import annotations

import json
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from super_agent.app.core.config import Settings

logger = logging.getLogger(__name__)

_MAX_BODY = 12_000


def email_ready(settings: "Settings") -> bool:
    if not settings.email_notifications_enabled:
        return False
    if not settings.smtp_user or not settings.smtp_password:
        return False
    return bool(settings.notification_email or settings.smtp_user)


def _recipient(settings: "Settings") -> str:
    return (settings.notification_email or settings.smtp_user or "").strip()


def _sender(settings: "Settings") -> str:
    return (settings.email_from or settings.smtp_user or "").strip()


def send_email(
    settings: "Settings",
    subject: str,
    body: str,
    *,
    html: str | None = None,
) -> bool:
    """
    Send one email via SMTP (TLS). Returns True if sent, False if skipped or failed.
    """
    if not email_ready(settings):
        return False
    to_addr = _recipient(settings)
    from_addr = _sender(settings)
    if not to_addr or not from_addr:
        return False

    text = body if len(body) <= _MAX_BODY else body[: _MAX_BODY - 20] + "\n\n…(truncated)"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject[:900]
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(text, "plain", "utf-8"))
    if html:
        msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            server.starttls(context=context)
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        logger.info("email sent: %s", subject[:80])
        return True
    except Exception:
        logger.exception("SMTP send failed (subject=%s)", subject[:60])
        return False


def notify_research_saved(
    settings: "Settings",
    *,
    title: str,
    report: str,
    source: str = "heartbeat",
) -> None:
    if not settings.notify_on_research:
        return
    snippet = report.strip()
    if len(snippet) > 4000:
        snippet = snippet[:3990] + "\n…"
    body = (
        f"Source: {source}\n"
        f"Title: {title}\n\n"
        f"--- Summary ---\n{snippet}\n"
    )
    send_email(settings, f"[Super Agent] New research: {title[:60]}", body)


def notify_background_task(
    settings: "Settings",
    *,
    kind: str,
    status: str,
    detail: str | None = None,
    error: str | None = None,
) -> None:
    if not settings.notify_on_background_tasks:
        return
    lines = [f"Task: {kind}", f"Status: {status}"]
    if detail:
        lines.append(f"Detail: {detail}")
    if error:
        lines.append(f"Error: {error}")
    send_email(
        settings,
        f"[Super Agent] Task {status}: {kind[:50]}",
        "\n".join(lines),
    )


def notify_chat_turn_complete(
    settings: "Settings",
    *,
    user_context: str,
    answer: str,
    route: str,
    intent: str,
) -> bool:
    """
    Email the user a copy of this chat turn. Returns True if SMTP accepted the message.
    """
    if not settings.notify_on_user_requested_email:
        logger.info("User requested email but SUPER_AGENT_NOTIFY_ON_USER_REQUESTED_EMAIL is off")
        return False
    if not email_ready(settings):
        logger.warning(
            "User requested email but SMTP is not configured. Set SUPER_AGENT_SMTP_USER, "
            "SUPER_AGENT_SMTP_PASSWORD, and optionally SUPER_AGENT_NOTIFICATION_EMAIL "
            "(or SMTP_USER / SMTP_PASSWORD / NOTIFICATION_EMAIL).",
        )
        return False
    body = (
        f"You asked to be emailed about this turn.\n\n"
        f"--- Your message ---\n{user_context[:2000]}\n\n"
        f"--- Agent ({route} / {intent}) ---\n{answer[:8000]}\n"
    )
    subj_hint = user_context.replace("\n", " ").strip()[:55]
    subject = f"[Super Agent] Chat: {subj_hint}" if subj_hint else "[Super Agent] Your requested chat reply"
    ok = send_email(settings, subject, body)
    if not ok:
        logger.error("User-requested email failed to send (see earlier SMTP error)")
    return ok


def notify_improvement(
    settings: "Settings",
    *,
    ok: bool,
    instruction: str,
    target_file: str,
    error: str | None = None,
) -> None:
    if not settings.notify_on_improve:
        return
    status = "succeeded" if ok else "failed"
    body = f"Instruction:\n{instruction[:2000]}\n\nTarget: {target_file}\n\nStatus: {status}\n"
    if error:
        body += f"\nError:\n{error[:2000]}\n"
    send_email(
        settings,
        f"[Super Agent] Code improvement {status}: {target_file[:40]}",
        body,
    )


def notify_agent_job_finished(
    settings: "Settings",
    *,
    job_id: str,
    prompt: str,
    error: str | None,
    answer: str | None,
) -> None:
    if not settings.notify_on_agent_job:
        return
    if error:
        body = f"Job: {job_id}\nPrompt:\n{prompt[:2000]}\n\nError:\n{error[:4000]}\n"
        sub = f"[Super Agent] Async job failed ({job_id[:8]})"
    else:
        body = f"Job: {job_id}\nPrompt:\n{prompt[:2000]}\n\n--- Reply ---\n{(answer or '')[:8000]}\n"
        sub = f"[Super Agent] Async job done ({job_id[:8]})"
    send_email(settings, sub, body)


def notify_full_stack_improvement(settings: "Settings", *, summary: dict[str, object]) -> None:
    if not settings.notify_on_improve:
        return
    body = json.dumps(summary, indent=2, default=str)[:_MAX_BODY]
    send_email(settings, "[Super Agent] Full-stack improvement finished", body)
