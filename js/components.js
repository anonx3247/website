// Shared components for the website
// This module provides reusable components that can be used across all pages

/**
 * Renders the site header with the name
 */
function renderHeader() {
    const header = document.createElement('h1');
    header.textContent = 'ANAS LECAILLON';
    document.body.insertBefore(header, document.body.firstChild);
}

/**
 * Renders the navigation bar
 */
function renderNavigation() {
    const nav = document.createElement('nav');
    nav.innerHTML = `
        <a href="index.html"><i class="fas fa-home icon"></i>Home</a>
        <a href="projects.html"><i class="fas fa-project-diagram icon"></i>Projects</a>
        <a href="experience.html"><i class="fas fa-briefcase icon"></i>Experience</a>
        <a href="education.html"><i class="fas fa-graduation-cap icon"></i>Education</a>
        <a href="skills.html"><i class="fas fa-tools icon"></i>Skills</a>
        <a href="https://msrchd.lecaillon.com"><i class="fas fa-flask icon"></i>Experiments</a>
    `;

    const firstH1 = document.querySelector('h1');
    if (firstH1 && firstH1.nextSibling) {
        firstH1.parentNode.insertBefore(nav, firstH1.nextSibling);
    } else if (firstH1) {
        firstH1.parentNode.appendChild(nav);
    } else {
        document.body.appendChild(nav);
    }
}

/**
 * Initialize common components
 * This should be called when the DOM is ready
 */
function initComponents() {
    renderHeader();
    renderNavigation();
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initComponents);
} else {
    initComponents();
}
