/**
 * Crawl Manager Module - Handles crawl submission and records display
 */
var CrawlManager = (function() {
    'use strict';

    var recordsPage = 1;
    var recordsPageSize = 20;

    /**
     * Initialize the crawl manager
     */
    function init() {
        // Bind crawl button
        var crawlBtn = document.getElementById('crawlBtn');
        if (crawlBtn) {
            crawlBtn.addEventListener('click', submitCrawl);
        }

        // Bind refresh records button
        var refreshRecordsBtn = document.getElementById('refreshCrawlRecords');
        if (refreshRecordsBtn) {
            refreshRecordsBtn.addEventListener('click', function() {
                loadRecords();
            });
        }

        // Bind status filter
        var statusFilter = document.getElementById('crawlStatusFilter');
        if (statusFilter) {
            statusFilter.addEventListener('change', function() {
                recordsPage = 1;
                loadRecords();
            });
        }

        // Setup HTML file upload
        setupHtmlUpload();
    }

    /**
     * Setup HTML file drag and drop upload
     */
    function setupHtmlUpload() {
        var dropZone = document.getElementById('htmlDropZone');
        var fileInput = document.getElementById('htmlFileInput');

        if (!dropZone || !fileInput) return;

        dropZone.addEventListener('click', function() {
            fileInput.click();
        });

        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', function() {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            var files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadHtmlFile(files[0]);
            }
        });

        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                uploadHtmlFile(fileInput.files[0]);
            }
        });
    }

    /**
     * Submit URLs for crawling
     */
    function submitCrawl() {
        var urlListEl = document.getElementById('urlList');
        var useProxyEl = document.getElementById('useProxy');
        var autoIndexEl = document.getElementById('autoIndex');
        var maxConcurrentEl = document.getElementById('maxConcurrent');
        var statusEl = document.getElementById('crawlStatus');

        if (!urlListEl || !statusEl) return;

        var urlText = urlListEl.value.trim();
        if (!urlText) {
            statusEl.innerHTML = '<div class="status-message error">Please enter at least one URL</div>';
            return;
        }

        var urls = urlText.split('\n')
            .map(function(u) { return u.trim(); })
            .filter(function(u) { return u.length > 0; });

        if (urls.length === 0) {
            statusEl.innerHTML = '<div class="status-message error">Please enter at least one valid URL</div>';
            return;
        }

        statusEl.innerHTML = '<div class="status-message loading"><span class="spinner"></span>Submitting crawl task...</div>';

        API.crawl.submitUrls({
            urls: urls,
            use_proxy: useProxyEl ? useProxyEl.checked : false,
            auto_index: autoIndexEl ? autoIndexEl.checked : true,
            max_concurrent: maxConcurrentEl ? parseInt(maxConcurrentEl.value, 10) || 5 : 5
        }).then(function(result) {
            statusEl.innerHTML = '<div class="status-message success">' +
                'Crawl task created successfully!<br>' +
                'Task ID: ' + result.task_id + '<br>' +
                'Total URLs: ' + result.total_urls + '<br>' +
                result.message +
            '</div>';

            // Clear the URL list
            urlListEl.value = '';

            // Refresh records after a short delay
            setTimeout(function() {
                loadRecords();
            }, 1000);
        }).catch(function(error) {
            statusEl.innerHTML = '<div class="status-message error">Error: ' +
                API.utils.escapeHtml(error.message) + '</div>';
        });
    }

    /**
     * Upload HTML file to extract links
     * @param {File} file - HTML file
     */
    function uploadHtmlFile(file) {
        var statusEl = document.getElementById('htmlUploadStatus');
        var useProxyEl = document.getElementById('htmlUseProxy');
        var autoIndexEl = document.getElementById('htmlAutoIndex');
        var autoCrawlEl = document.getElementById('htmlAutoCrawl');

        if (!statusEl) return;

        // Validate file type
        var ext = file.name.split('.').pop().toLowerCase();
        if (ext !== 'html' && ext !== 'htm') {
            statusEl.innerHTML = '<div class="status-message error">Please select an HTML file</div>';
            return;
        }

        statusEl.innerHTML = '<div class="status-message loading"><span class="spinner"></span>Uploading and parsing HTML...</div>';

        API.crawl.uploadHtml(file, {
            use_proxy: useProxyEl ? useProxyEl.checked : false,
            auto_index: autoIndexEl ? autoIndexEl.checked : true,
            auto_crawl: autoCrawlEl ? autoCrawlEl.checked : true
        }).then(function(result) {
            var urlPreview = '';
            if (result.extracted_urls && result.extracted_urls.length > 0) {
                urlPreview = '<br><strong>Sample URLs:</strong><ul style="margin: 10px 0; padding-left: 20px;">';
                result.extracted_urls.slice(0, 5).forEach(function(url) {
                    urlPreview += '<li style="word-break: break-all; font-size: 0.85rem;">' +
                        API.utils.escapeHtml(API.utils.truncate(url, 60)) + '</li>';
                });
                if (result.extracted_urls.length > 5) {
                    urlPreview += '<li>... and ' + (result.extracted_urls.length - 5) + ' more</li>';
                }
                urlPreview += '</ul>';
            }

            statusEl.innerHTML = '<div class="status-message success">' +
                result.message + '<br>' +
                'Total URLs extracted: ' + result.total_urls +
                (result.task_id ? '<br>Task ID: ' + result.task_id : '') +
                urlPreview +
            '</div>';

            // Refresh records after a short delay
            if (result.task_id) {
                setTimeout(function() {
                    loadRecords();
                }, 1000);
            }
        }).catch(function(error) {
            statusEl.innerHTML = '<div class="status-message error">Error: ' +
                API.utils.escapeHtml(error.message) + '</div>';
        });
    }

    /**
     * Load crawl records from API
     */
    function loadRecords() {
        var container = document.getElementById('crawlRecordsList');
        if (!container) return;

        var statusFilter = document.getElementById('crawlStatusFilter');
        var status = statusFilter ? statusFilter.value : '';

        container.innerHTML = '<p class="placeholder"><span class="spinner"></span>Loading crawl records...</p>';

        API.crawl.getRecords({
            page: recordsPage,
            page_size: recordsPageSize,
            status: status || undefined
        }).then(function(data) {
            renderRecords(data);
        }).catch(function(error) {
            container.innerHTML = '<div class="error-message">Error loading crawl records: ' +
                API.utils.escapeHtml(error.message) + '</div>';
        });
    }

    /**
     * Render crawl records list
     * @param {Object} data - Records data from API
     */
    function renderRecords(data) {
        var container = document.getElementById('crawlRecordsList');
        if (!container) return;

        if (!data.records || data.records.length === 0) {
            container.innerHTML = '<p class="placeholder">No crawl records found</p>';
            renderRecordsPagination(data.total, data.page, data.page_size);
            return;
        }

        var html = data.records.map(function(record) {
            var statusClass = 'status-' + record.status;

            return '<div class="crawl-record-item">' +
                '<div class="crawl-record-info">' +
                    '<div class="url">' +
                        '<a href="' + API.utils.escapeHtml(record.url) + '" target="_blank" style="color:#4facfe">' +
                            API.utils.truncate(API.utils.escapeHtml(record.url), 80) +
                        '</a>' +
                    '</div>' +
                    '<div class="record-meta">' +
                        '<span class="status-badge ' + statusClass + '">' + record.status + '</span>' +
                        (record.title ? ' | ' + API.utils.truncate(API.utils.escapeHtml(record.title), 40) : '') +
                        (record.content_length ? ' | ' + formatExtractedTextLength(record.content_length) : '') +
                        (record.task_id ? ' | Task #' + record.task_id : '') +
                        ' | ' + API.utils.formatRelativeTime(record.created_at) +
                    '</div>' +
                    (record.error_message ?
                        '<div style="color:#f44336; font-size:0.8rem; margin-top:4px">Error: ' +
                            API.utils.truncate(API.utils.escapeHtml(record.error_message), 60) + '</div>' : '') +
                '</div>' +
            '</div>';
        }).join('');

        container.innerHTML = html;
        renderRecordsPagination(data.total, data.page, data.page_size);
    }

    /**
     * Format extracted text length for display.
     * @param {number} length - Character count
     * @returns {string}
     */
    function formatExtractedTextLength(length) {
        if (!length) return 'Extracted 0 chars';
        return 'Extracted ' + String(length) + ' chars';
    }

    /**
     * Render records pagination controls
     * @param {number} total - Total items
     * @param {number} page - Current page
     * @param {number} size - Page size
     */
    function renderRecordsPagination(total, page, size) {
        var container = document.getElementById('crawlRecordsPagination');
        if (!container) return;

        var totalPages = Math.ceil(total / size);
        if (totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        var html = '';

        // Previous button
        html += '<button ' + (page <= 1 ? 'disabled' : '') +
            ' onclick="CrawlManager.goToRecordsPage(' + (page - 1) + ')">Prev</button>';

        // Page numbers
        var startPage = Math.max(1, page - 2);
        var endPage = Math.min(totalPages, page + 2);

        for (var i = startPage; i <= endPage; i++) {
            html += '<button class="' + (i === page ? 'active' : '') +
                '" onclick="CrawlManager.goToRecordsPage(' + i + ')">' + i + '</button>';
        }

        // Next button
        html += '<button ' + (page >= totalPages ? 'disabled' : '') +
            ' onclick="CrawlManager.goToRecordsPage(' + (page + 1) + ')">Next</button>';

        container.innerHTML = html;
    }

    /**
     * Navigate to a specific records page
     * @param {number} page - Page number
     */
    function goToRecordsPage(page) {
        recordsPage = page;
        loadRecords();
    }

    // Public API
    return {
        init: init,
        loadRecords: loadRecords,
        goToRecordsPage: goToRecordsPage
    };
})();
