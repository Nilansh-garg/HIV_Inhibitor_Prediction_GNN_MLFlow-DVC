/* ============================================================
   MolPredict — Contact page JS
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const submitBtn   = document.getElementById('submit-btn');
  const submitText  = document.getElementById('submit-text');
  const submitIcon  = document.getElementById('submit-icon');
  const form        = document.getElementById('contact-form');
  const successDiv  = document.getElementById('form-success');
  const resetBtn    = document.getElementById('reset-form-btn');

  const fields = {
    name:    document.getElementById('f-name'),
    email:   document.getElementById('f-email'),
    subject: document.getElementById('f-subject'),
    message: document.getElementById('f-message'),
  };

  function validate() {
    let valid = true;

    Object.values(fields).forEach(f => f.classList.remove('error'));

    if (!fields.name.value.trim()) {
      fields.name.classList.add('error'); valid = false;
    }

    const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRe.test(fields.email.value.trim())) {
      fields.email.classList.add('error'); valid = false;
    }

    if (!fields.subject.value) {
      fields.subject.classList.add('error'); valid = false;
    }

    if (fields.message.value.trim().length < 10) {
      fields.message.classList.add('error'); valid = false;
    }

    return valid;
  }

  submitBtn.addEventListener('click', async () => {
    if (!validate()) return;

    submitBtn.disabled = true;
    submitText.textContent = 'Sending…';
    submitIcon.textContent = '…';

    // Simulate async send (replace with real endpoint if needed)
    await new Promise(r => setTimeout(r, 1200));

    form.style.display        = 'none';
    successDiv.style.display  = 'flex';
  });

  resetBtn.addEventListener('click', () => {
    Object.values(fields).forEach(f => { f.value = ''; f.classList.remove('error'); });
    successDiv.style.display  = 'none';
    form.style.display        = 'block';
    submitBtn.disabled        = false;
    submitText.textContent    = 'Send Message';
    submitIcon.textContent    = '→';
  });

  // Live clear error on input
  Object.values(fields).forEach(f => {
    f.addEventListener('input', () => f.classList.remove('error'));
    f.addEventListener('change', () => f.classList.remove('error'));
  });
});
