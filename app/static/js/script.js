'use strict';

// Modal functionality
// Variables for modal elements
const modal = document.querySelector('[data-modal]');
const modalCloseBtn = document.querySelector('[data-modal-close]');
const modalCloseOverlay = document.querySelector('[data-modal-overlay]');

// Close modal function
if (modal && modalCloseBtn && modalCloseOverlay) {
  const modalCloseFunc = function () {
    modal.classList.add('closed'); // Add 'closed' class to hide modal
  };

  // Event listeners to close the modal
  modalCloseOverlay.addEventListener('click', modalCloseFunc);
  modalCloseBtn.addEventListener('click', modalCloseFunc);
}

// Notification toast functionality
// Variables for notification toast elements
const notificationToast = document.querySelector('[data-toast]');
const toastCloseBtn = document.querySelector('[data-toast-close]');

// Close notification toast
if (notificationToast && toastCloseBtn) {
  toastCloseBtn.addEventListener('click', function () {
    notificationToast.classList.add('closed'); // Add 'closed' class to hide toast
  });
}

// Mobile menu functionality
// Variables for mobile menu elements
const mobileMenuOpenBtn = document.querySelectorAll('[data-mobile-menu-open-btn]');
const mobileMenu = document.querySelectorAll('[data-mobile-menu]');
const mobileMenuCloseBtn = document.querySelectorAll('[data-mobile-menu-close-btn]');
const overlay = document.querySelector('[data-overlay]');

// Handle multiple mobile menus
if (mobileMenuOpenBtn && mobileMenu && mobileMenuCloseBtn && overlay) {
  for (let i = 0; i < mobileMenuOpenBtn.length; i++) {
    // Function to close mobile menu and overlay
    const mobileMenuCloseFunc = function () {
      mobileMenu[i].classList.remove('active'); // Remove 'active' class from menu
      overlay.classList.remove('active'); // Remove 'active' class from overlay
    };

    // Open mobile menu and activate overlay
    mobileMenuOpenBtn[i].addEventListener('click', function () {
      mobileMenu[i].classList.add('active'); // Add 'active' class to menu
      overlay.classList.add('active'); // Add 'active' class to overlay
    });

    // Close mobile menu and deactivate overlay
    mobileMenuCloseBtn[i].addEventListener('click', mobileMenuCloseFunc);
    overlay.addEventListener('click', mobileMenuCloseFunc);
  }
}

const mobileMenuOpenBtnAi = document.querySelectorAll('[data-mobile-menu-open-btn-Ai]');
const mobileMenuAi = document.querySelectorAll('[data-mobile-menu-Ai]');
const mobileMenuCloseBtnAi = document.querySelectorAll('[data-mobile-menu-close-btn-Ai]');
const overlayAi = document.querySelector('[data-overlay]');

// Handle multiple mobile menus
if (mobileMenuOpenBtnAi && mobileMenuAi && mobileMenuCloseBtnAi && overlayAi) {
  for (let i = 0; i < mobileMenuOpenBtnAi.length; i++) {
    // Function to close mobile menu and overlay
    const mobileMenuCloseFunc = function () {
      mobileMenuAi[i].classList.remove('active'); // Remove 'active' class from menu
      overlayAi.classList.remove('active'); // Remove 'active' class from overlay
    };

    // Open mobile menu and activate overlay
    mobileMenuOpenBtnAi[i].addEventListener('click', function () {
      mobileMenuAi[i].classList.add('active'); // Add 'active' class to menu
      overlayAi.classList.add('active'); // Add 'active' class to overlay
    });

    // Close mobile menu and deactivate overlay
    mobileMenuCloseBtnAi[i].addEventListener('click', mobileMenuCloseFunc);
    overlayAi.addEventListener('click', mobileMenuCloseFunc);
  }
}

const laptopMenuOpenBtnAi = document.querySelectorAll('[data-laptop-menu-open-btn-Ai]');
const laptopMenuAi = document.querySelectorAll('[data-mobile-menu-Ai]');
const laptopMenuCloseBtnAi = document.querySelectorAll('[data-mobile-menu-close-btn-Ai]');
const laptopOverlayAi = document.querySelector('[data-overlay]');

// Handle multiple mobile menus
if (laptopMenuOpenBtnAi && laptopMenuAi && laptopMenuCloseBtnAi && laptopOverlayAi) {
  for (let i = 0; i < laptopMenuOpenBtnAi.length; i++) {
    // Function to close mobile menu and overlay
    const mobileMenuCloseFunc = function () {
      laptopMenuAi[i].classList.remove('active'); // Remove 'active' class from menu
      laptopOverlayAi.classList.remove('active'); // Remove 'active' class from overlay
    };

    // Open mobile menu and activate overlay
    laptopMenuOpenBtnAi[i].addEventListener('click', function () {
      laptopMenuAi[i].classList.add('active'); // Add 'active' class to menu
      laptopOverlayAi.classList.add('active'); // Add 'active' class to overlay
    });

    // Close mobile menu and deactivate overlay
    laptopMenuCloseBtnAi[i].addEventListener('click', mobileMenuCloseFunc);
    laptopOverlayAi.addEventListener('click', mobileMenuCloseFunc);
  }
}
// Accordion functionality
// Variables for accordion elements
const accordionBtn = document.querySelectorAll('[data-accordion-btn]');
const accordion = document.querySelectorAll('[data-accordion]');

// Handle accordion toggle
if (accordionBtn && accordion) {
  for (let i = 0; i < accordionBtn.length; i++) {
    accordionBtn[i].addEventListener('click', function () {
      const clickedBtn = this.nextElementSibling.classList.contains('active'); // Check if clicked accordion is active

      // Close all accordions if another is clicked
      for (let i = 0; i < accordion.length; i++) {
        if (clickedBtn) break;
        if (accordion[i].classList.contains('active')) {
          accordion[i].classList.remove('active'); // Remove 'active' class from accordion content
          accordionBtn[i].classList.remove('active'); // Remove 'active' class from accordion button
        }
      }

      // Toggle the 'active' class for the clicked accordion
      this.nextElementSibling.classList.toggle('active');
      this.classList.toggle('active');
    });
  }
}

// Remove default 'title' attribute from all ion-icons
document.querySelectorAll('ion-icon').forEach((icon) => {
  icon.removeAttribute('title'); // Remove default title attribute to prevent browser tooltips
});