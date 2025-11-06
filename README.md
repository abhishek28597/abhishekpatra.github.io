# Karpathy-Style Blog

A minimalist, academic-style blog theme for GitHub Pages, inspired by Andrej Karpathy's blog.

## Features

- ✨ Clean, minimalist design focused on readability
- 📝 Optimized for technical writing and code snippets
- 🎨 Syntax highlighting for code blocks
- 📐 LaTeX math support via MathJax
- 📱 Fully responsive design
- 🚀 Fast loading times
- 🔍 SEO optimized
- 📊 RSS feed support

## Quick Start

### 1. Fork and Setup Repository

1. Create a new repository named `[your-username].github.io`
2. Copy all files from this blog template to your repository
3. Update `_config.yml` with your information:
   - `title`: Your name or blog title
   - `email`: Your email
   - `description`: Blog description
   - `url`: `https://[your-username].github.io`
   - `twitter_username`: Your Twitter handle
   - `github_username`: Your GitHub username

### 2. Local Development (Optional)

If you want to test locally before pushing to GitHub:

```bash
# Install Ruby and Bundler (if not already installed)
# On macOS:
brew install ruby
gem install bundler

# On Ubuntu:
sudo apt-get install ruby-full build-essential
gem install bundler

# Install Jekyll and dependencies
bundle install

# Run locally
bundle exec jekyll serve

# View your site at http://localhost:4000
```

### 3. Writing Posts

Create new posts in the `_posts` directory with the format:
`YYYY-MM-DD-title-of-post.md`

Example front matter:

```yaml
---
layout: post
title: "Your Post Title"
date: 2024-11-06
categories: [category1, category2]
tags: [tag1, tag2, tag3]
excerpt: "A brief description of your post that appears in the index."
reading_time: 5
---

Your content here...
```

### 4. Deploy to GitHub Pages

1. Push all files to your repository:
```bash
git add .
git commit -m "Initial blog setup"
git push origin main
```

2. Enable GitHub Pages:
   - Go to Settings → Pages
   - Source: Deploy from branch
   - Branch: main, / (root)
   - Save

Your blog will be live at `https://[your-username].github.io` within a few minutes!

## Customization

### Styling

Edit `assets/css/style.css` to customize:
- Colors
- Fonts
- Spacing
- Layout widths

### Adding Pages

Create new pages (like About, Projects, etc.) in the root directory:

```markdown
---
layout: default
title: Projects
permalink: /projects/
---

# Projects

Your content here...
```

### Navigation

Edit the navigation links in `_layouts/default.html`:

```html
<nav class="site-nav">
  <a href="{{ '/' | relative_url }}">Blog</a>
  <a href="{{ '/about' | relative_url }}">About</a>
  <a href="{{ '/projects' | relative_url }}">Projects</a>
  <!-- Add more links here -->
</nav>
```

## Writing Tips

### Code Blocks

Use triple backticks with language specification:

\`\`\`python
def hello_world():
    print("Hello, World!")
\`\`\`

### Mathematics

Use LaTeX syntax:
- Inline: `$e^{i\pi} + 1 = 0$`
- Display: `$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$`

### Images

Store images in `assets/images/` and reference them:

```markdown
![Alt text](/assets/images/your-image.png)
```

## File Structure

```
.
├── _config.yml          # Site configuration
├── _layouts/            # Page templates
│   ├── default.html     # Main layout
│   └── post.html        # Post layout
├── _posts/              # Blog posts
├── assets/              
│   ├── css/            
│   │   └── style.css    # Main stylesheet
│   └── images/          # Image storage
├── about.md             # About page
├── archive.md           # Archive page
├── index.html           # Homepage
├── Gemfile              # Ruby dependencies
└── README.md            # This file
```

## Advanced Features

### Drafts

Create drafts in a `_drafts` folder. They won't be published until moved to `_posts`.

### Custom Domain

1. Create a `CNAME` file with your domain
2. Configure DNS settings with your domain provider
3. Enable HTTPS in GitHub Pages settings

### Comments

Add a commenting system like Disqus or utterances by adding the embed code to `_layouts/post.html`.

### Analytics

Add Google Analytics by inserting the tracking code in `_layouts/default.html` before `</head>`.

## Troubleshooting

### Page not updating?
- Check GitHub Actions for build errors
- Clear browser cache
- Wait a few minutes for changes to propagate

### Local build errors?
- Run `bundle update` to update dependencies
- Make sure you have the correct Ruby version
- Check for syntax errors in YAML front matter

## License

MIT License - feel free to use this template for your own blog!

## Credits

Design inspired by [Andrej Karpathy's blog](https://karpathy.github.io/).
