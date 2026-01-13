# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a static personal portfolio website for Anas Lecaillon, hosted at **anas.lecaillon.com**. No build system or package manager is used.

## Development

To preview the site locally, open `index.html` in a browser or use a local server:
```bash
python -m http.server 8000
```

## Architecture

**Page Structure:**
- `index.html` - Home page with profile, social links, and GitHub contribution graph
- `projects.html` - Project showcase in a responsive grid layout
- `experience.html` - Work experience timeline
- `education.html` - Education timeline
- `skills.html` - Skills display with interactive hover effects

**Styling:**
- `index.css` - All styles for the site (dark theme, #141414 background)
- `font.otf` - Custom font for headings (CustomFont)
- Uses Google Fonts: Cinzel, Crimson Text
- Uses Font Awesome 6.4.0 for icons

**Key CSS Components:**
- `.timeline` / `.timeline-item` - Vertical timeline with left border for education/experience
- `.experience-card` - Card layout for work experience entries
- `.project-grid` / `.project-card` - Responsive grid for projects
- `.skills-container` / `.skill` - Flexbox pill layout for skills

**Shared Navigation:**
Each HTML page includes the same navigation bar manually. When modifying navigation, update all 5 HTML files.

## Deployment

After pushing changes to the repo, deploy to production:
```bash
ssh root@lecaillon.com
cd ~/website
git pull
cp -r * /var/www/anas/
```

The server runs nginx and serves files from `/var/www/anas/`.
