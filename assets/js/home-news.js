// Home page "Latest News" renderer — auto-populated from assets/data/posts.json.
// Adding a new blog? Just append an entry to posts.json; this section + blog.html update automatically.
(function () {
  'use strict';
  const el = document.getElementById('home-news-list');
  if (!el) return;

  const LIMIT = 3;
  let allPosts = [];

  function getLang() {
    try {
      const saved = localStorage.getItem('lf-lang');
      if (saved === 'zh' || saved === 'en') return saved;
    } catch (e) { }
    return (navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[c]);
  }

  function primaryTag(p, lang) {
    const tags = (lang === 'zh' ? p.tags_zh : p.tags_en) || p.tags_en || [];
    return tags[0] || '';
  }

  function render() {
    const lang = getLang();
    const posts = allPosts.slice(0, LIMIT);
    el.innerHTML = posts.map(p => {
      const loc = p[lang] || p.en;
      const tag = primaryTag(p, lang);
      return `<a href="${escapeHtml(loc.url)}" class="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 card-hover block">
        <div class="text-xs text-gray-500 mb-2">${escapeHtml(p.date)}${tag ? ' · <span>' + escapeHtml(tag) + '</span>' : ''}</div>
        <div class="font-semibold mb-1">${escapeHtml(loc.title)}</div>
        <div class="text-sm text-gray-600 dark:text-gray-400">${escapeHtml(loc.summary)}</div>
      </a>`;
    }).join('');
  }

  fetch('assets/data/posts.json?v=20260510', { cache: 'no-cache' })
    .then(r => r.json())
    .then(posts => {
      allPosts = posts.sort((a, b) => (a.date < b.date ? 1 : -1));
      render();
    })
    .catch(err => {
      el.innerHTML = `<p class="text-red-500 col-span-3">Failed to load posts: ${err}</p>`;
    });

  window.addEventListener('lf:langchange', render);
})();
