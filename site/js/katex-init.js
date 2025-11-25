document.addEventListener("DOMContentLoaded", function() {
  if (typeof renderMathInElement === 'undefined') {
    console.warn('KaTeX auto-render not loaded');
    return;
  }
  renderMathInElement(document.body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false},
      {left: '\\(', right: '\\)', display: false},
      {left: '\\[', right: '\\]', display: true}
    ],
    throwOnError: false
  });
});
