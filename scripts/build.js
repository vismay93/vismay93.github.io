const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

// Configure marked to ignore math blocks so they don't get processed as markdown.
const blockMath = {
    name: 'blockMath',
    level: 'block',
    start(src) { return src.indexOf('$$'); },
    tokenizer(src, tokens) {
        const match = src.match(/^\$\$([\s\S]+?)\$\$/);
        if (match) {
            return {
                type: 'blockMath',
                raw: match[0],
                math: match[1]
            };
        }
    },
    renderer(token) {
        return `$$${token.math}$$`;
    }
};

const inlineMath = {
    name: 'inlineMath',
    level: 'inline',
    start(src) { return src.indexOf('$'); },
    tokenizer(src, tokens) {
        const match = src.match(/^\$([^\$\s](?:[^\$]*?[^\$\s])?)\$/);
        if (match) {
            return {
                type: 'inlineMath',
                raw: match[0],
                math: match[1]
            };
        }
    },
    renderer(token) {
        return `$${token.math}$`;
    }
};

marked.use({ extensions: [blockMath, inlineMath] });

// Path setup
const rootDir = path.join(__dirname, '..');
const blogDir = path.join(rootDir, 'blog');
const postsJsonPath = path.join(blogDir, 'posts.json');
const postTemplatePath = path.join(blogDir, 'post.html');

console.log('Starting static blog post compilation...');

// 1. Read post metadata
if (!fs.existsSync(postsJsonPath)) {
    console.error(`Error: posts.json not found at ${postsJsonPath}`);
    process.exit(1);
}
const posts = JSON.parse(fs.readFileSync(postsJsonPath, 'utf8'));

// 2. Read template
if (!fs.existsSync(postTemplatePath)) {
    console.error(`Error: post.html template not found at ${postTemplatePath}`);
    process.exit(1);
}
const templateHtml = fs.readFileSync(postTemplatePath, 'utf8');

// List to collect all page URLs for the sitemap
const sitemapUrls = [
    'https://vismay93.github.io/index.html',
    'https://vismay93.github.io/blog.html'
];

// 3. Compile each post
posts.forEach(post => {
    const mdPath = path.join(blogDir, 'posts', `${post.slug}.md`);
    if (!fs.existsSync(mdPath)) {
        console.warn(`Warning: Markdown file not found for slug "${post.slug}" at ${mdPath}. Skipping.`);
        return;
    }

    console.log(`Compiling post: ${post.title} (${post.slug})`);

    const mdContent = fs.readFileSync(mdPath, 'utf8');
    const compiledContentHtml = marked.parse(mdContent);

    // Dynamic date formatting
    const formattedDate = new Date(post.date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });

    // We replace metadata in the template
    let postHtml = templateHtml;

    // A. Title Tag
    postHtml = postHtml.replace(
        '<title>Blog Post — Vismay Patel</title>',
        () => `<title>${post.title} — Vismay Patel</title>`
    );

    // B. Meta Description
    postHtml = postHtml.replace(
        '<meta name="description" content="Blog post by Vismay Patel">',
        () => `<meta name="description" content="${post.excerpt}">`
    );

    // C. Open Graph, Twitter Card & Canonical link insertion
    const seoMeta = `
    <!-- SEO Meta Tags -->
    <meta property="og:title" content="${post.title} — Vismay Patel">
    <meta property="og:description" content="${post.excerpt}">
    <meta property="og:image" content="https://vismay93.github.io/${post.image}">
    <meta property="og:url" content="https://vismay93.github.io/blog/${post.slug}.html">
    <meta property="og:type" content="article">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="${post.title} — Vismay Patel">
    <meta name="twitter:description" content="${post.excerpt}">
    <meta name="twitter:image" content="https://vismay93.github.io/${post.image}">
    <link rel="canonical" href="https://vismay93.github.io/blog/${post.slug}.html">`;

    // Insert SEO meta right before the closing </head> tag or after existing meta tags
    postHtml = postHtml.replace('</head>', () => `${seoMeta}\n</head>`);

    // D. Header Post Title
    postHtml = postHtml.replace(
        '<h1 id="postTitle" style="font-size: 2.6rem; max-width: 740px; margin: 0 auto;">Loading...</h1>',
        () => `<h1 id="postTitle" style="font-size: 2.6rem; max-width: 740px; margin: 0 auto;">${post.title}</h1>`
    );

    // E. Meta info header
    const postMetaHtml = `<div class="blog-meta" style="justify-content: center; margin-bottom: 16px;" id="postMeta">` +
        `<span class="blog-category">${post.category}</span>` +
        `<span>${formattedDate}</span>` +
        `<span>${post.readTime} read</span>` +
        `<span style="display: inline-flex; align-items: center; gap: 6px;" id="postViews"></span>` +
        `</div>`;
    postHtml = postHtml.replace(
        /<div class="blog-meta"[^>]*id="postMeta"[^>]*><\/div>/,
        () => postMetaHtml
    );

    // F. Pre-rendered HTML Body
    const postBodyHtml = `<div class="blog-post-body" id="postBody">
                ${compiledContentHtml}
            </div>`;
    postHtml = postHtml.replace(
        /<div class="blog-post-body[^>]*id="postBody"[^>]*>([\s\S]*?)<\/div>/,
        () => postBodyHtml
    );

    // G. Remove marked script resource, since it's pre-rendered
    postHtml = postHtml.replace(
        '<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>',
        () => ''
    );

    // H. Replace client-side script loader block with the simplified static version
    const simplifiedScript = `<script>
        (async function () {
            const slug = "${post.slug}";

            // Add copy buttons to code snippets
            document.querySelectorAll('.blog-post-body pre').forEach((pre) => {
                const button = document.createElement('button');
                button.className = 'copy-code-btn';
                button.innerHTML = '<i class="fa fa-copy"></i>';
                button.setAttribute('aria-label', 'Copy code snippet');
                
                button.addEventListener('click', async () => {
                    const code = pre.querySelector('code');
                    const codeText = code ? code.innerText : pre.innerText;
                    try {
                        await navigator.clipboard.writeText(codeText);
                        button.innerHTML = '<i class="fa fa-check" style="color: #2cb67d;"></i>';
                        button.classList.add('copied');
                        setTimeout(() => {
                            button.innerHTML = '<i class="fa fa-copy"></i>';
                            button.classList.remove('copied');
                        }, 1500);
                    } catch (err) {
                        console.error('Failed to copy code: ', err);
                    }
                });
                
                pre.appendChild(button);
            });

            // Load views badge dynamically (using komarev.com - completely free and login-free!)
            const viewsImg = document.createElement('img');
            const counterId = \`https://vismay93.github.io/blog/post.html?slug=\${slug}\`;
            viewsImg.src = \`https://api.visitorbadge.io/api/visitors?path=\${counterId}&labelColor=%232d3a8c&countColor=%231a7a6d&label=Views\`;
            viewsImg.alt = '';
            viewsImg.style.height = '20px';
            viewsImg.style.borderRadius = '4px';
            viewsImg.style.verticalAlign = 'middle';
            const viewsContainer = document.getElementById('postViews');
            if (viewsContainer) {
                viewsContainer.appendChild(viewsImg);
            }

            // Initialize Giscus Comments & Reactions
            const giscusScript = document.createElement('script');
            giscusScript.src = 'https://giscus.app/client.js';
            giscusScript.setAttribute('data-repo', 'vismay93/vismay93.github.io');
            giscusScript.setAttribute('data-repo-id', 'MDEwOlJlcG9zaXRvcnkzMjUwNDg0NjY=');
            giscusScript.setAttribute('data-category', 'General'); 
            giscusScript.setAttribute('data-category-id', 'DIC_kwDOE1_Yks4C9r-Q');
            giscusScript.setAttribute('data-mapping', 'pathname');
            giscusScript.setAttribute('data-strict', '0');
            giscusScript.setAttribute('data-reactions-enabled', '1');
            giscusScript.setAttribute('data-emit-metadata', '0');
            giscusScript.setAttribute('data-input-position', 'bottom');
            giscusScript.setAttribute('data-theme', 'noborder_light');
            giscusScript.setAttribute('data-lang', 'en');
            giscusScript.crossOrigin = 'anonymous';
            giscusScript.async = true;
            document.body.appendChild(giscusScript);

            // Trigger MathJax typesetting if MathJax is loaded
            if (window.MathJax) {
                if (window.MathJax.typesetPromise) {
                    window.MathJax.typesetPromise([document.getElementById('postBody')]).catch(err => console.log(err));
                } else if (window.MathJax.startup && window.MathJax.startup.promise) {
                    window.MathJax.startup.promise.then(() => {
                        window.MathJax.typesetPromise([document.getElementById('postBody')]);
                    }).catch(err => console.log(err));
                }
            }
        })();
    </script>`;

    // Replace the script starting from (async function () to matching final script tag
    const scriptStartRegex = /<script>\s*\/\/ Configure marked extensions[\s\S]*?<\/script>/;
    postHtml = postHtml.replace(scriptStartRegex, () => simplifiedScript);

    // Save pre-rendered file to blog/
    const outputPath = path.join(blogDir, `${post.slug}.html`);
    fs.writeFileSync(outputPath, postHtml, 'utf8');
    console.log(`Successfully generated static post at ${outputPath}`);

    // Add to sitemap URLs
    sitemapUrls.push(`https://vismay93.github.io/blog/${post.slug}.html`);
});

// 4. Generate sitemap.xml
console.log('Generating sitemap.xml...');
const sitemapXml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemapUrls.map(url => `  <url>
    <loc>${url}</loc>
    <changefreq>${url.includes('.html') && !url.includes('blog.html') ? 'monthly' : 'weekly'}</changefreq>
    <priority>${url.includes('index.html') ? '1.0' : url.includes('blog.html') ? '0.8' : '0.7'}</priority>
  </url>`).join('\n')}
</urlset>`;

const sitemapPath = path.join(rootDir, 'sitemap.xml');
fs.writeFileSync(sitemapPath, sitemapXml, 'utf8');
console.log(`Successfully generated sitemap.xml at ${sitemapPath}`);

// 5. Generate robots.txt
console.log('Generating robots.txt...');
const robotsTxt = `User-agent: *
Allow: /

Sitemap: https://vismay93.github.io/sitemap.xml
`;
const robotsPath = path.join(rootDir, 'robots.txt');
fs.writeFileSync(robotsPath, robotsTxt, 'utf8');
console.log(`Successfully generated robots.txt at ${robotsPath}`);

console.log('Blog compilation build process complete!');
