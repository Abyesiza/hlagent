"use client";

import { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  char: string;
  speed: number;
  opacity: number;
  size: number;
  drift: number;       // horizontal drift rate
  phase: number;       // breathing phase offset
  color: string;
}

const CHARS = "0123456789ABCDEF01×∑∂∇λαβγδε√∞≈≤≥±";
const COLORS = [
  "61,130,255",   // --blue
  "0,212,255",    // --cyan
  "0,184,144",    // --teal
  "139,92,246",   // --violet
  "55,66,101",    // --txt-3 (muted)
];

function rand(a: number, b: number) { return a + Math.random() * (b - a); }
function pick<T>(arr: T[]): T { return arr[Math.floor(Math.random() * arr.length)]!; }

function makeParticle(W: number, H: number): Particle {
  return {
    x:       rand(0, W),
    y:       rand(-H, H),
    char:    pick(CHARS.split("")),
    speed:   rand(0.18, 0.8),
    opacity: rand(0.06, 0.28),
    size:    rand(9, 18),
    drift:   rand(-0.12, 0.12),
    phase:   rand(0, Math.PI * 2),
    color:   pick(COLORS),
  };
}

export default function DataBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = window.innerWidth;
    let H = window.innerHeight;
    let raf: number;
    let t = 0;

    // Density: ~1 particle per 4 000 px²
    const makeParticles = (): Particle[] => {
      const count = Math.max(60, Math.floor((W * H) / 4_000));
      return Array.from({ length: count }, () => makeParticle(W, H));
    };

    canvas.width  = W;
    canvas.height = H;
    let particles = makeParticles();

    // Occasionally swap a character so the field "pulses"
    const swapInterval = setInterval(() => {
      const p = particles[Math.floor(Math.random() * particles.length)];
      if (p) p.char = pick(CHARS.split(""));
    }, 80);

    const onResize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width  = W;
      canvas.height = H;
      particles = makeParticles();
    };
    window.addEventListener("resize", onResize);

    const draw = () => {
      t += 0.008;
      ctx.clearRect(0, 0, W, H);

      for (const p of particles) {
        // Breathing: opacity oscillates with a sine wave
        const breathe = 0.55 + 0.45 * Math.sin(t * 0.9 + p.phase);
        const alpha = p.opacity * breathe;

        ctx.font = `${p.size}px "JetBrains Mono", monospace`;
        ctx.fillStyle = `rgba(${p.color},${alpha.toFixed(3)})`;
        ctx.fillText(p.char, p.x, p.y);

        // Float upward + gentle horizontal drift
        p.y   -= p.speed;
        p.x   += p.drift * Math.sin(t + p.phase);

        // Wrap: when particle floats off-top, reset at bottom
        if (p.y < -p.size) {
          p.y    = H + p.size;
          p.x    = rand(0, W);
          p.char = pick(CHARS.split(""));
        }
        // Wrap horizontal
        if (p.x < -20) p.x = W + 10;
        if (p.x > W + 20) p.x = -10;
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      clearInterval(swapInterval);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
        opacity: 1,
      }}
      aria-hidden="true"
    />
  );
}
