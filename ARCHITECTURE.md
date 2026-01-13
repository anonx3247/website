# Website Architecture

This document describes the modular architecture of the portfolio website.

## Overview

The website is a **static HTML/CSS/JavaScript site** with a modular component-based architecture. No build system or package manager is required.

## Directory Structure

```
.
├── index.html              # Home page
├── projects.html           # Projects showcase
├── experience.html         # Work experience timeline
├── education.html          # Education timeline
├── skills.html             # Skills with interactive tooltips
├── index.css              # Global styles
├── js/                    # JavaScript modules
│   ├── components.js      # Shared header and navigation components
│   └── tooltip.js         # Skill tooltip functionality
└── data/                  # JSON data files
    └── skills.json        # Skill data for tooltips
```

## Modular Components

### 1. Shared Components (`js/components.js`)

This module provides reusable components that are automatically loaded on all pages:

- **Header**: Renders the main "ANAS LECAILLON" title
- **Navigation**: Renders the navigation bar with links to all pages

**Usage**: Include this script in the `<head>` of any page:
```html
<script src="js/components.js"></script>
```

The components are automatically initialized when the DOM is ready.

### 2. Tooltip System (`js/tooltip.js` + `data/skills.json`)

The tooltip system provides interactive hover effects for skills:

- **`js/tooltip.js`**: Module that handles tooltip rendering and positioning
- **`data/skills.json`**: Centralized data file containing skill relationships

**Usage**: Include both scripts on pages that need tooltips:
```html
<script src="js/components.js"></script>
<script src="js/tooltip.js"></script>
```

## Benefits of This Architecture

### 1. **Single Source of Truth**
- Navigation and header are defined once in `js/components.js`
- Skill data is centralized in `data/skills.json`
- Changes to navigation or skill data only need to be made in one place

### 2. **Easy Maintenance**
- To update navigation: edit `js/components.js`
- To add/modify skills: edit `data/skills.json`
- No need to update multiple HTML files

### 3. **Separation of Concerns**
- HTML pages contain only page-specific content
- Shared UI components are in JavaScript modules
- Data is separated from presentation logic

### 4. **No Build System Required**
- Pure HTML/CSS/JavaScript
- Can be hosted on any static file server
- No compilation or bundling needed

## Making Changes

### Adding a Navigation Link

Edit `js/components.js` in the `renderNavigation()` function:

```javascript
function renderNavigation() {
    const nav = document.createElement('nav');
    nav.innerHTML = `
        <a href="index.html"><i class="fas fa-home icon"></i>Home</a>
        <!-- Add new links here -->
    `;
    // ...
}
```

### Adding Skill Data

Edit `data/skills.json`:

```json
{
    "New Skill": {
        "projects": [
            { "name": "Project Name", "url": "project-url" }
        ],
        "experience": [
            { "name": "Experience Name", "url": "experience-url" }
        ],
        "education": [
            { "name": "Education Name", "url": "education-url" }
        ]
    }
}
```

### Creating a New Page

1. Create a new HTML file (e.g., `newpage.html`)
2. Include the components script in the `<head>`:
   ```html
   <script src="js/components.js"></script>
   ```
3. Add your page-specific content in the `<body>`
4. The header and navigation will be automatically added

## Browser Compatibility

The modules use modern JavaScript features:
- ES6+ syntax (arrow functions, const/let, template literals)
- Fetch API (for loading JSON data)
- Async/await

All modern browsers are supported. For older browsers, transpilation would be needed.

## Local Development

To preview the site locally:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Deployment

After making changes:

1. Test locally
2. Commit and push to repository
3. Deploy to production server:
   ```bash
   ssh root@lecaillon.com
   cd ~/website
   git pull
   cp -r * /var/www/anas/
   ```
