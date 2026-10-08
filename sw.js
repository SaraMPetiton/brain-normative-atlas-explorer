// Keeps a copy of the app on the device so it works offline.
// Network first: when online, the latest files are used (e.g., a new normative_model.js),
// and the saved copy is refreshed.
// When offline, the saved copy is used.
const CACHE = 'brain-normative-atlas-v3';
const FILES = ['./', './index.html', './normative_model.js', './manifest.webmanifest', './icons/smiley.jpg', './icons/icon-192.png', './icons/icon-512.png', './icons/icon-180.png', './icons/icon-64.png', './icons/icon-maskable-512.png'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(k => Promise.all(k.filter(n => n !== CACHE).map(n => caches.delete(n)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== self.location.origin) return;
  e.respondWith(fetch(e.request).then(r => { if (r.ok) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); } return r; })
    .catch(() => caches.match(e.request, { ignoreSearch: true })));
});
