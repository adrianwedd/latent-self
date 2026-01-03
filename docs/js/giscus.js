// Inject Giscus comment widget
window.addEventListener('load', () => {
  const article = document.querySelector('main article');
  if (!article) return;
  const container = document.createElement('div');
  container.id = 'giscus-comments';
  article.appendChild(container);

  const s = document.createElement('script');
  s.src = 'https://giscus.app/client.js';
  s.async = true;
  s.crossOrigin = 'anonymous';
  s.setAttribute('data-repo', 'adrianwedd/latent-self');
  s.setAttribute('data-repo-id', 'MDEwOlJlcG9zaXRvcnk0NDc2NzE1NDI=');
  s.setAttribute('data-category', 'General');
  s.setAttribute('data-category-id', 'DIC_kwDOHeU3AM4B_w3l');
  s.setAttribute('data-mapping', 'pathname');
  s.setAttribute('data-reactions-enabled', '1');
  s.setAttribute('data-emit-metadata', '0');
  s.setAttribute('data-theme', 'light');
  container.appendChild(s);
});
