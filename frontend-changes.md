# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a dark/light theme toggle button that persists the user's preference via `localStorage`.

## Files Modified

### `frontend/style.css`
- **Global transition**: Added `background-color`, `border-color`, and `color` transitions to `*` so all elements animate smoothly on theme switch.
- **Light theme variables**: Added `[data-theme="light"]` block on the `html` element with:
  - `--background: #f8fafc` (near-white)
  - `--surface: #ffffff`
  - `--surface-hover: #f1f5f9`
  - `--text-primary: #0f172a` (dark slate for contrast)
  - `--text-secondary: #475569`
  - `--border-color: #e2e8f0`
  - `--assistant-message: #e2e8f0`
  - `--shadow` reduced opacity for light backgrounds
  - `--welcome-bg: #dbeafe`
- **`.theme-toggle` button styles**: Fixed position (`top: 1rem; right: 1rem`), circular (40×40px), uses `--surface`/`--border-color` variables, hover scales up and highlights with primary color.
- **Sun/moon icon transitions**: `.icon-sun` visible in dark mode (opacity 1, no rotation); `.icon-moon` visible in light mode. Each icon cross-fades and rotates on toggle.

### `frontend/index.html`
- Added `<button class="theme-toggle" id="themeToggle">` just before the `<script>` tags, containing inline SVG sun and moon icons with `aria-label` and `title` for accessibility.
- Bumped CSS/JS cache-bust version from `v=11` to `v=12`.

### `frontend/script.js`
- Added `initTheme()` called in `DOMContentLoaded`:
  - Reads `localStorage.getItem('theme')` on load and applies `data-theme="light"` to `document.documentElement` if set.
  - Attaches click listener to `#themeToggle`.
- Added `toggleTheme()`:
  - Checks current `data-theme` attribute on `<html>`.
  - Removes attribute (dark) or sets `data-theme="light"` (light), then persists to `localStorage`.
