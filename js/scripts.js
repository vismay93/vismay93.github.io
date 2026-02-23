/*!
 * Portfolio — Modern Scripts
 * Vanilla JS (no jQuery dependency)
 */

(function () {
  'use strict';

  // ------- Scroll-triggered animations -------
  const animateElements = document.querySelectorAll('.animate-on-scroll, .animate-scale');

  const observerOptions = {
    root: null,
    rootMargin: '0px 0px -60px 0px',
    threshold: 0.15
  };

  const animationObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        animationObserver.unobserve(entry.target);
      }
    });
  }, observerOptions);

  animateElements.forEach((el) => animationObserver.observe(el));

  // ------- Navbar scroll effect -------
  const navbar = document.getElementById('navbar');

  function handleNavbarScroll() {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }

  window.addEventListener('scroll', handleNavbarScroll, { passive: true });
  handleNavbarScroll(); // run on load

  // ------- Active nav link on scroll -------
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

  function highlightActiveNav() {
    const scrollY = window.scrollY + 120;

    sections.forEach((section) => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      const sectionId = section.getAttribute('id');

      if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
        navLinks.forEach((link) => {
          link.classList.remove('active');
          if (link.getAttribute('href') === '#' + sectionId) {
            link.classList.add('active');
          }
        });
      }
    });
  }

  window.addEventListener('scroll', highlightActiveNav, { passive: true });

  // ------- Smooth scroll for nav links -------
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === '#') return;

      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        const offset = 70;
        const top = target.getBoundingClientRect().top + window.pageYOffset - offset;
        window.scrollTo({ top, behavior: 'smooth' });

        // Close mobile menu if open
        const navLinksEl = document.getElementById('navLinks');
        const navToggleEl = document.getElementById('navToggle');
        if (navLinksEl.classList.contains('active')) {
          navLinksEl.classList.remove('active');
          navToggleEl.classList.remove('active');
          document.body.style.overflow = '';
        }
      }
    });
  });

  // ------- Mobile menu toggle -------
  const navToggle = document.getElementById('navToggle');
  const navLinksMenu = document.getElementById('navLinks');

  if (navToggle && navLinksMenu) {
    navToggle.addEventListener('click', () => {
      navToggle.classList.toggle('active');
      navLinksMenu.classList.toggle('active');

      if (navLinksMenu.classList.contains('active')) {
        document.body.style.overflow = 'hidden';
      } else {
        document.body.style.overflow = '';
      }
    });
  }

  // ------- Hero scroll indicator -------
  const scrollIndicator = document.querySelector('.hero-scroll-indicator');
  if (scrollIndicator) {
    scrollIndicator.addEventListener('click', () => {
      const hero = document.getElementById('hero');
      if (hero && hero.nextElementSibling) {
        const main = hero.nextElementSibling;
        const firstSection = main.querySelector('section') || main;
        const top = firstSection.getBoundingClientRect().top + window.pageYOffset - 70;
        window.scrollTo({ top, behavior: 'smooth' });
      }
    });
  }

  // ------- Stat counter animation -------
  const statNumbers = document.querySelectorAll('.stat-number');

  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const finalText = el.textContent;
        const match = finalText.match(/(\d+)/);
        if (match) {
          const target = parseInt(match[1], 10);
          const suffix = finalText.replace(match[1], '');
          let current = 0;
          const duration = 1500;
          const step = Math.max(1, Math.floor(target / (duration / 30)));
          const interval = setInterval(() => {
            current += step;
            if (current >= target) {
              current = target;
              clearInterval(interval);
            }
            el.textContent = current + suffix;
          }, 30);
        }
        counterObserver.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  statNumbers.forEach((el) => counterObserver.observe(el));

})();
