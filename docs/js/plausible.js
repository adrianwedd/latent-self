// Track search queries using Plausible
if (window.plausible) {
  document.addEventListener('DOMContentLoaded', function () {
    var form = document.querySelector('form.md-search__form');
    if (form) {
      form.addEventListener('submit', function () {
        var input = document.querySelector('input.md-search__input');
        if (input && input.value) {
          plausible('Search', {props: {query: input.value}});
        }
      });
    }
  });
}
