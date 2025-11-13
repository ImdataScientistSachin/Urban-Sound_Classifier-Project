/**
 * Animations for Urban Sound Classifier
 * Enhances UI with modern animations and interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const header = document.querySelector('header');
    const uploadArea = document.querySelector('.upload-area');
    const formatBadges = document.querySelectorAll('.format-badge');
    const soundClassItems = document.querySelectorAll('.sound-class-item');
    const logoContainer = document.querySelector('.logo-container');
    
    // Add entrance animations
    animateEntrance();
    
    // Add scroll animations
    window.addEventListener('scroll', () => {
        const scrollPosition = window.scrollY;
        
        // Subtle parallax effect on header
        if (header) {
            header.style.backgroundPositionY = `${scrollPosition * 0.2}px`;
        }
        
        // Add subtle shadow to header on scroll
        if (scrollPosition > 10) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
    
    // Animate format badges on hover
    if (formatBadges.length > 0) {
        formatBadges.forEach((badge, index) => {
            badge.style.animationDelay = `${index * 0.1}s`;
            
            badge.addEventListener('mouseenter', () => {
                badge.classList.add('badge-pulse');
            });
            
            badge.addEventListener('mouseleave', () => {
                setTimeout(() => {
                    badge.classList.remove('badge-pulse');
                }, 300);
            });
        });
    }
    
    // Animate sound class items
    if (soundClassItems.length > 0) {
        soundClassItems.forEach((item, index) => {
            item.style.animationDelay = `${index * 0.05}s`;
        });
    }
    
    // Add logo hover animation
    if (logoContainer) {
        logoContainer.addEventListener('mouseenter', () => {
            logoContainer.classList.add('logo-pulse');
        });
        
        logoContainer.addEventListener('mouseleave', () => {
            setTimeout(() => {
                logoContainer.classList.remove('logo-pulse');
            }, 300);
        });
    }
});

/**
 * Animate elements on page load
 */
function animateEntrance() {
    // Header elements
    const headerTitle = document.querySelector('header h1');
    const tagline = document.querySelector('.tagline');
    const formatBadges = document.querySelector('.format-badges');
    
    // Main content elements
    const uploadContainer = document.querySelector('.upload-container');
    const resultsContainer = document.querySelector('.results-container');
    const micButton = document.querySelector('.mic-button');
    const micModal = document.querySelector('.mic-modal');
    const visualizer = document.querySelector('.visualizer');
    
    // Footer elements
    const footerSections = document.querySelectorAll('.footer-section');
    
    // Add animation classes with delays
    if (headerTitle) {
        headerTitle.classList.add('fade-in-up');
        headerTitle.style.animationDelay = '0.2s';
    }
    
    if (tagline) {
        tagline.classList.add('fade-in-up');
        tagline.style.animationDelay = '0.4s';
    }
    
    if (formatBadges) {
        formatBadges.classList.add('fade-in-up');
        formatBadges.style.animationDelay = '0.6s';
    }
    
    if (uploadContainer) {
        uploadContainer.classList.add('fade-in-up');
        uploadContainer.style.animationDelay = '0.8s';
    }
    
    if (resultsContainer) {
        resultsContainer.classList.add('fade-in-up');
        resultsContainer.style.animationDelay = '1s';
    }
    
    if (micButton) {
        micButton.classList.add('fade-in-up');
        micButton.style.animationDelay = '0.9s';
    }
    
    if (footerSections.length > 0) {
        footerSections.forEach((section, index) => {
            section.classList.add('fade-in-up');
            section.style.animationDelay = `${1.2 + (index * 0.2)}s`;
        });
    }
}

/**
 * Add animation classes to CSS
 */
function addAnimationStyles() {
    const styleSheet = document.createElement('style');
    styleSheet.type = 'text/css';
    styleSheet.innerText = `
        .fade-in-up {
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.6s ease-out forwards;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .badge-pulse {
            animation: badgePulse 0.6s ease-out;
        }
        
        @keyframes badgePulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        .logo-pulse {
            animation: logoPulse 0.6s ease-out;
        }
        
        @keyframes logoPulse {
            0% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.1) rotate(5deg); }
            100% { transform: scale(1) rotate(0deg); }
        }
        
        .mic-modal {
            opacity: 0;
            transform: scale(0.9);
            animation: modalFadeIn 0.3s ease-out forwards;
        }
        
        @keyframes modalFadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        
        .visualizer-bar {
            animation: visualizerPulse 1.2s ease-in-out infinite;
        }
        
        @keyframes visualizerPulse {
            0% { transform: scaleY(0.1); }
            50% { transform: scaleY(1); }
            100% { transform: scaleY(0.1); }
        }
    `;
    document.head.appendChild(styleSheet);
}

// Add animation styles when DOM is loaded
document.addEventListener('DOMContentLoaded', addAnimationStyles);