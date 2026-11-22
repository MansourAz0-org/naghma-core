// Semicircular cents meter drawn on a canvas at display refresh rate.
// `read()` is polled every animation frame (e.g. from a LiveStore); React
// never re-renders for needle motion.

import { useEffect, useRef } from "react";

import { METER_RANGE_CENTS, needleAngle, springStep, tuningZone } from "./tuning";

const MAX_ANGLE = (Math.PI * 5) / 12; // ±75°

const COLORS = {
  inTune: "#e9c46a",
  close: "#4fd1c5",
  off: "#ef7a6b",
  idle: "rgba(255,255,255,0.18)",
  tick: "rgba(255,255,255,0.28)",
  tickMajor: "rgba(255,255,255,0.6)",
};

export interface CentsMeterProps {
  /** Current deviation in cents, or `null` when no note is held. */
  read: () => number | null;
  flatLabel: string;
  sharpLabel: string;
  className?: string;
}

export function CentsMeter({ read, flatLabel, sharpLabel, className = "cents-meter" }: CentsMeterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const readRef = useRef(read);
  readRef.current = read;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let last = performance.now();
    let value = 0;
    let velocity = 0;
    let presence = 0; // 0..1 fade for needle when no note
    /** What the canvas shows; null = it must be redrawn. */
    let shown: { value: number; presence: number; w: number; h: number } | null = null;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      shown = null; // Resizing clears the canvas.
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const cents = readRef.current();
      const note = cents !== null;
      const target = cents ?? 0;
      [value, velocity] = springStep(value, velocity, target, dt, note ? 16 : 6);
      presence += ((note ? 1 : 0) - presence) * Math.min(1, dt * 8);

      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      raf = requestAnimationFrame(draw);
      // A still needle (held note or idle) looks the same: skip the redraw.
      if (
        shown &&
        shown.w === w &&
        shown.h === h &&
        Math.abs(shown.value - value) < 0.01 &&
        tuningZone(shown.value) === tuningZone(value) &&
        Math.abs(shown.presence - presence) < 0.002
      )
        return;
      shown = { value, presence, w, h };
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2;
      const cy = h * 0.92;
      const r = Math.min(w * 0.46, h * 0.85);

      // Arc track.
      ctx.lineCap = "round";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.beginPath();
      ctx.arc(cx, cy, r, -Math.PI / 2 - MAX_ANGLE, -Math.PI / 2 + MAX_ANGLE);
      ctx.stroke();

      // In-tune zone (±5 cents).
      const zone = needleAngle(5, MAX_ANGLE);
      ctx.lineWidth = 10;
      ctx.strokeStyle = `rgba(233,196,106,${0.15 + 0.35 * presence * (tuningZone(value) === "inTune" ? 1 : 0.4)})`;
      ctx.beginPath();
      ctx.arc(cx, cy, r, -Math.PI / 2 - zone, -Math.PI / 2 + zone);
      ctx.stroke();

      // Ticks every 5 cents; major at 0, ±25, ±50.
      for (let c = -METER_RANGE_CENTS; c <= METER_RANGE_CENTS; c += 5) {
        const a = -Math.PI / 2 + needleAngle(c, MAX_ANGLE);
        const major = c % 25 === 0;
        const len = major ? 16 : 8;
        ctx.strokeStyle = major ? COLORS.tickMajor : COLORS.tick;
        ctx.lineWidth = major ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * (r - 14), cy + Math.sin(a) * (r - 14));
        ctx.lineTo(cx + Math.cos(a) * (r - 14 - len), cy + Math.sin(a) * (r - 14 - len));
        ctx.stroke();
      }

      // Labels.
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.font = "500 13px system-ui, sans-serif";
      ctx.textAlign = "center";
      for (const c of [-50, -25, 25, 50]) {
        const a = -Math.PI / 2 + needleAngle(c, MAX_ANGLE);
        const label = c > 0 ? `+${c}` : `−${-c}`;
        ctx.fillText(label, cx + Math.cos(a) * (r - 46), cy + Math.sin(a) * (r - 46) + 4);
      }
      ctx.font = "600 14px system-ui, sans-serif";
      const fa = -Math.PI / 2 - MAX_ANGLE - 0.12;
      const sa = -Math.PI / 2 + MAX_ANGLE + 0.12;
      ctx.fillText(flatLabel, cx + Math.cos(fa) * r, cy + Math.sin(fa) * r + 18);
      ctx.fillText(sharpLabel, cx + Math.cos(sa) * r, cy + Math.sin(sa) * r + 18);

      // Needle.
      const zoneName = tuningZone(value);
      const color = presence < 0.05 ? COLORS.idle : COLORS[zoneName];
      const a = -Math.PI / 2 + needleAngle(value, MAX_ANGLE);
      ctx.globalAlpha = 0.25 + 0.75 * presence;
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.shadowColor = color;
      ctx.shadowBlur = 18 * presence;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a) * (r - 6), cy + Math.sin(a) * (r - 6));
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [flatLabel, sharpLabel]);

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />;
}
