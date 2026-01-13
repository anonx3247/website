// Tooltip module for skill hover interactions
// Loads skill data from JSON and displays tooltips on hover

let skillData = {};

/**
 * Loads skill data from JSON file
 */
async function loadSkillData() {
    try {
        const response = await fetch('data/skills.json');
        skillData = await response.json();
    } catch (error) {
        console.error('Failed to load skill data:', error);
    }
}

/**
 * Creates tooltip element if it doesn't exist
 */
function createTooltipElement() {
    let tooltip = document.getElementById('skillTooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'skillTooltip';
        tooltip.className = 'skill-tooltip';
        document.body.appendChild(tooltip);

        // Keep tooltip visible when hovering over it
        tooltip.addEventListener('mouseenter', () => {
            tooltip.classList.add('visible');
        });

        tooltip.addEventListener('mouseleave', () => {
            tooltip.classList.remove('visible');
        });
    }
    return tooltip;
}

/**
 * Builds tooltip content HTML
 */
function buildTooltipContent(data) {
    let content = '';

    if (data.projects && data.projects.length > 0) {
        const projectLinks = data.projects.map(p =>
            `<a href="${p.url}" class="tooltip-link">${p.name}</a>`
        ).join('');
        content += `<div class="tooltip-section">
            <div class="tooltip-label">Projects</div>
            <div class="tooltip-items">${projectLinks}</div>
        </div>`;
    }

    if (data.experience && data.experience.length > 0) {
        const expLinks = data.experience.map(p =>
            `<a href="${p.url}" class="tooltip-link">${p.name}</a>`
        ).join('');
        content += `<div class="tooltip-section">
            <div class="tooltip-label">Experience</div>
            <div class="tooltip-items">${expLinks}</div>
        </div>`;
    }

    if (data.education && data.education.length > 0) {
        const eduLinks = data.education.map(p =>
            `<a href="${p.url}" class="tooltip-link">${p.name}</a>`
        ).join('');
        content += `<div class="tooltip-section">
            <div class="tooltip-label">Education</div>
            <div class="tooltip-items">${eduLinks}</div>
        </div>`;
    }

    return content;
}

/**
 * Positions tooltip relative to skill element
 */
function positionTooltip(tooltip, skillElement) {
    const rect = skillElement.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) + 'px';
    tooltip.style.top = (rect.bottom + 10) + 'px';
}

/**
 * Initialize tooltip functionality for all skill elements
 */
function initTooltips() {
    const tooltip = createTooltipElement();
    const skills = document.querySelectorAll('.skill');

    skills.forEach(skill => {
        skill.addEventListener('mouseenter', (e) => {
            const skillName = skill.textContent;
            const data = skillData[skillName];

            if (data) {
                const content = buildTooltipContent(data);
                if (content) {
                    tooltip.innerHTML = content;
                    positionTooltip(tooltip, skill);
                    tooltip.classList.add('visible');
                }
            }
        });

        skill.addEventListener('mouseleave', () => {
            tooltip.classList.remove('visible');
        });
    });
}

/**
 * Initialize tooltip system
 * Loads data and sets up event listeners
 */
async function initTooltipSystem() {
    await loadSkillData();
    initTooltips();
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTooltipSystem);
} else {
    initTooltipSystem();
}
