# Complete Codebase Explanation for Beginners

## What is This?

This is a **Jekyll blog** - a static website generator that turns Markdown files into a beautiful blog. It's designed to be hosted on **GitHub Pages** (free hosting). Think of it like a template that automatically creates web pages from your writing.

---

## How Jekyll Works (Simple Explanation)

1. You write blog posts in **Markdown** (a simple text format)
2. Jekyll reads your files and configuration
3. It combines your content with templates (layouts)
4. It generates HTML files (web pages)
5. GitHub Pages hosts these files for free

---

## File Structure Explained

### 📁 Root Directory Files

#### `_config.yml` ⚙️
**What it does:** The main configuration file - like settings for your entire blog.

**What to change:**
- `title`: Your name or blog title (appears in header)
- `email`: Your email address
- `description`: What your blog is about (for search engines)
- `url`: Your website URL (e.g., `https://abhishek28597.github.io`)
- `baseurl`: Usually `/your-username.github.io` or leave empty
- `twitter_username`: Your Twitter handle (or remove if you don't use it)
- `github_username`: Your GitHub username

**When to edit:** First thing you should do! Personalize this with your info.

---

#### `index.html` 🏠
**What it does:** Your homepage - shows a list of all your blog posts.

**What to change:** Usually you don't need to change this. It automatically displays all posts from `_posts/` folder.

**How it works:** Uses Jekyll's Liquid templating to loop through all posts and display them.

---

#### `about.md` 👤
**What it does:** Your "About" page - tells visitors who you are.

**What to change:** 
- Replace all the placeholder text with your actual information
- Update your background, interests, contact info
- Add links to your social media profiles

**Format:** Markdown file with front matter (the `---` section at top)

---

#### `archive.md` 📚
**What it does:** Shows all your posts organized by year.

**What to change:** Usually no changes needed - it automatically lists all posts.

---

#### `Gemfile` 📦
**What it does:** Lists all Ruby dependencies (plugins) your blog needs.

**What to change:** Only if you want to add new Jekyll plugins. For beginners, leave it alone.

**Note:** This is for Ruby/Jekyll. You run `bundle install` to install these.

---

#### `README.md` 📖
**What it does:** Documentation/instructions for your blog.

**What to change:** You can update it with your own instructions or leave it as is.

---

### 📁 `_layouts/` Directory

This folder contains **templates** - the structure that wraps around your content.

#### `default.html` 🎨
**What it does:** The main template for all pages. Defines:
- HTML structure (header, navigation, footer)
- Links to CSS and fonts
- Navigation menu
- Math rendering (MathJax) and code highlighting scripts

**What to change:**
- **Navigation links:** Edit the `<nav>` section (around line 30-40) to add/remove menu items
- **Header title:** The site title link (already uses `site.title` from `_config.yml`)
- **Footer:** The copyright text at bottom
- **Add analytics:** Insert Google Analytics code before `</head>` if you want tracking

**Key sections:**
- `<head>`: Meta tags, CSS, fonts, scripts
- `<header>`: Site title and navigation
- `<main>`: Where your page content goes (`{{ content }}`)
- `<footer>`: Copyright info

---

#### `post.html` 📝
**What it does:** Template specifically for blog post pages. Shows:
- Post title and date
- Post content
- Tags
- Previous/Next post navigation

**What to change:**
- Usually no changes needed
- If you want to add comments (Disqus, utterances), add the code in the footer section
- If you want to change how tags are displayed, edit the tags section

---

### 📁 `_posts/` Directory

**What it does:** This is where all your blog posts live!

**Naming format:** `YYYY-MM-DD-title-of-post.md`
- Example: `2024-11-06-understanding-attention.md`
- **Important:** Must follow this exact format!

**Post structure:**
Each post has two parts:

1. **Front Matter** (between `---` lines):
```yaml
---
layout: post
title: "Your Post Title"
date: 2024-11-06
categories: [category1, category2]
tags: [tag1, tag2]
excerpt: "Short description"
reading_time: 5
---
```

2. **Content** (below front matter):
   - Write in Markdown
   - Use `#` for headings
   - Use `**bold**` for bold text
   - Use code blocks with triple backticks

**What to change:**
- Create new `.md` files here for new posts
- Always use the date format in filename
- Fill in the front matter with your post info

---

### 📁 `assets/css/` Directory

#### `style.css` 🎨
**What it does:** All the styling/design for your blog - colors, fonts, spacing, layout.

**What to change:**
- **Colors:** Search for color codes like `#0066cc` (blue links) and change them
- **Fonts:** Change font families (currently Merriweather and Source Sans Pro)
- **Spacing:** Adjust margins and padding values
- **Width:** Change `max-width: 700px` in `.container` to make content wider/narrower
- **Font sizes:** Adjust `font-size` values throughout

**Key sections:**
- `.container`: Overall page width and padding
- `.site-title`: Blog title styling
- `.post-content`: How blog post text looks
- `@media` queries: Mobile responsive design

---

## Common Tasks & What Files to Edit

### ✏️ Writing a New Blog Post

1. Create a new file in `_posts/` folder
2. Name it: `YYYY-MM-DD-your-title.md`
3. Add front matter at top:
```yaml
---
layout: post
title: "Your Title Here"
date: 2024-12-01
categories: [your-category]
tags: [tag1, tag2]
excerpt: "Brief description"
---
```
4. Write your content below in Markdown
5. Save and push to GitHub

---

### 🎨 Changing Colors

Edit `assets/css/style.css`:
- Link color: Search for `#0066cc` and replace
- Text color: Search for `#333` (dark gray) and replace
- Background: Search for `background-color: #fff` and replace

---

### 📝 Updating About Page

Edit `about.md`:
- Replace placeholder text with your info
- Update contact links
- Add your background/experience

---

### 🔧 Personalizing Site Info

Edit `_config.yml`:
- Change `title` to your name
- Update `email`, `url`, `description`
- Add/remove social media usernames

---

### ➕ Adding Navigation Links

Edit `_layouts/default.html`:
- Find the `<nav class="site-nav">` section
- Add new links like:
```html
<a href="{{ '/projects' | relative_url }}">Projects</a>
```

---

### 🖼️ Adding Images

1. Create `assets/images/` folder (if it doesn't exist)
2. Put your images there
3. Reference in posts:
```markdown
![Alt text](/assets/images/your-image.png)
```

---

## How Everything Works Together

1. **Jekyll reads `_config.yml`** → Gets site settings
2. **Jekyll reads `_posts/*.md`** → Gets your blog posts
3. **Jekyll uses `_layouts/post.html`** → Wraps each post in the post template
4. **Jekyll uses `_layouts/default.html`** → Wraps everything in the main template
5. **Jekyll applies `style.css`** → Styles everything
6. **Jekyll generates `index.html`** → Creates homepage with post list
7. **GitHub Pages hosts it** → Makes it live on the web

---

## File Priority (What Overrides What)

- **`_config.yml`** → Site-wide settings
- **Layout files** → Page structure
- **CSS file** → Visual styling
- **Post front matter** → Individual post settings
- **Post content** → Your actual writing

---

## Quick Reference: File → Purpose

| File | Purpose | Change Frequency |
|------|---------|------------------|
| `_config.yml` | Site settings | Once (initial setup) |
| `index.html` | Homepage | Rarely |
| `about.md` | About page | Once (personalize) |
| `archive.md` | Archive page | Never |
| `_layouts/default.html` | Main template | Occasionally |
| `_layouts/post.html` | Post template | Rarely |
| `_posts/*.md` | Blog posts | Every new post |
| `assets/css/style.css` | Styling | As needed |
| `Gemfile` | Dependencies | Rarely |

---

## Tips for Beginners

1. **Start with `_config.yml`** - Personalize your site info first
2. **Edit `about.md`** - Make it yours
3. **Write a test post** - Create a simple post to see how it works
4. **Experiment with CSS** - Change colors to see what happens
5. **Use Markdown** - Learn basic Markdown syntax (it's simple!)
6. **Test locally** - Run `bundle exec jekyll serve` to preview changes

---

## Need Help?

- **Jekyll docs:** https://jekyllrb.com/docs/
- **Markdown guide:** https://www.markdownguide.org/
- **GitHub Pages:** https://pages.github.com/

---

## Summary

- **Configuration:** `_config.yml`
- **Templates:** `_layouts/`
- **Posts:** `_posts/`
- **Styling:** `assets/css/style.css`
- **Pages:** `about.md`, `archive.md`, `index.html`

Most of your time will be spent:
1. Writing posts in `_posts/`
2. Occasionally tweaking `style.css`
3. Updating `_config.yml` and `about.md` when needed

That's it! You now understand your entire blog codebase! 🎉

