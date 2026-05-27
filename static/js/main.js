/* ══════════════════════════════════════
   SALARY IQ  ·  main.js
══════════════════════════════════════ */

// ── ANIMATED BACKGROUND ──────────────────────────────────────────────────────

(function initCanvas() {
  const canvas = document.getElementById('bg-canvas');
  const ctx    = canvas.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeParticle() {
    return {
      x:    Math.random() * W,
      y:    Math.random() * H,
      r:    Math.random() * 1.2 + 0.3,
      vx:   (Math.random() - 0.5) * 0.18,
      vy:   (Math.random() - 0.5) * 0.18,
      alpha: Math.random() * 0.5 + 0.1,
    };
  }

  function initParticles() {
    particles = Array.from({ length: 90 }, makeParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 229, 160, ${p.alpha})`;
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });

    // subtle grid
    ctx.strokeStyle = 'rgba(0, 229, 160, 0.025)';
    ctx.lineWidth   = 1;
    const gap = 80;
    for (let x = 0; x < W; x += gap) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += gap) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', () => { resize(); initParticles(); });
  resize();
  initParticles();
  draw();
})();

// ── SLIDER LIVE VALUES ────────────────────────────────────────────────────────

const sliders = [
  { id: 'experience_years', badge: 'exp-val',    suffix: ' yrs' },
  { id: 'skills_count',     badge: 'skills-val', suffix: ''     },
  { id: 'certifications',   badge: 'cert-val',   suffix: ''     },
];

sliders.forEach(({ id, badge, suffix }) => {
  const input = document.getElementById(id);
  const span  = document.getElementById(badge);

  function update() {
    span.textContent = input.value + suffix;
    // track fill
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.background = `linear-gradient(to right, #00e5a0 ${pct}%, rgba(255,255,255,0.07) ${pct}%)`;
  }

  input.addEventListener('input', update);
  update();
});

// ── HEALTH CHECK ─────────────────────────────────────────────────────────────

(async function healthCheck() {
  const dot  = document.getElementById('health-dot');
  const text = document.getElementById('health-text');
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    if (data.status === 'ok' && data.model_loaded) {
      dot.className  = 'dot dot--ok';
      text.textContent = 'Model online';
    } else {
      dot.className  = 'dot dot--error';
      text.textContent = 'Model not loaded — place .pkl in Best_model/';
    }
  } catch {
    dot.className  = 'dot dot--error';
    text.textContent = 'Service unreachable';
  }
})();

// ── FORM SUBMIT ───────────────────────────────────────────────────────────────

const form         = document.getElementById('salary-form');
const predictBtn   = document.getElementById('predict-btn');
const resultPanel  = document.getElementById('result-panel');
const resultAmount = document.getElementById('result-amount');
const errorPanel   = document.getElementById('error-panel');
const errorMsg     = document.getElementById('error-msg');
const resetBtn     = document.getElementById('reset-btn');

function showError(msg) {
  errorPanel.classList.remove('hidden');
  errorMsg.textContent = msg;
}

function hideError() {
  errorPanel.classList.add('hidden');
}

function getFormData() {
  const fd = new FormData(form);
  return {
    job_title:        fd.get('job_title')       || '',
    industry:         fd.get('industry')         || '',
    location:         fd.get('location')         || '',
    remote_work:      fd.get('remote_work')      || '',
    education_level:  fd.get('education_level')  || '',
    company_size:     fd.get('company_size')      || '',
    experience_years: Number(fd.get('experience_years') || 0),
    skills_count:     Number(fd.get('skills_count')     || 0),
    certifications:   Number(fd.get('certifications')   || 0),
  };
}

function validate(data) {
  const required = ['job_title', 'industry', 'location', 'remote_work', 'education_level', 'company_size'];
  for (const key of required) {
    if (!data[key]) return `Please fill in: ${key.replace(/_/g, ' ')}`;
  }
  return null;
}

// Animated number counter
function animateNumber(target, duration = 800) {
  const start = performance.now();
  const update = (now) => {
    const t    = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const val  = Math.round(ease * target);
    resultAmount.textContent = '$' + val.toLocaleString();
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();

  const data = getFormData();
  const err  = validate(data);
  if (err) { showError(err); return; }

  predictBtn.classList.add('loading');
  predictBtn.querySelector('.btn-label').textContent = 'Predicting…';

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(data),
    });
    const json = await res.json();

    if (!res.ok || json.error) {
      showError(json.error || 'Prediction failed.');
    } else {
      resultPanel.classList.remove('hidden');
      resultPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
      animateNumber(json.salary);
    }
  } catch {
    showError('Network error — is the server running?');
  } finally {
    predictBtn.classList.remove('loading');
    predictBtn.querySelector('.btn-label').textContent = 'Predict Salary';
  }
});

resetBtn.addEventListener('click', () => {
  resultPanel.classList.add('hidden');
  form.reset();
  sliders.forEach(({ id, badge, suffix }) => {
    const input = document.getElementById(id);
    const span  = document.getElementById(badge);
    input.value      = input.min;
    span.textContent = input.min + suffix;
    input.style.background = `linear-gradient(to right, #00e5a0 0%, rgba(255,255,255,0.07) 0%)`;
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
