/**
 * Documents Manager Module - Handles document list display and actions
 */
var DocumentsManager = (function() {
    'use strict';

    var currentPage = 1;
    var pageSize = 20;

    /**
     * Initialize the documents manager
     */
    function init() {
        // Bind event listeners
        var refreshBtn = document.getElementById('refreshDocs');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function() {
                loadDocuments();
            });
        }

        var sourceTypeFilter = document.getElementById('sourceTypeFilter');
        if (sourceTypeFilter) {
            sourceTypeFilter.addEventListener('change', function() {
                currentPage = 1;
                loadDocuments();
            });
        }

        var dataSourceFilter = document.getElementById('dataSourceFilter');
        if (dataSourceFilter) {
            dataSourceFilter.addEventListener('change', function() {
                currentPage = 1;
                loadDocuments();
            });
        }
    }

    /**
     * Load documents from API
     */
    function loadDocuments() {
        var container = document.getElementById('docsList');
        if (!container) return;

        var sourceTypeFilter = document.getElementById('sourceTypeFilter');
        var dataSourceFilter = document.getElementById('dataSourceFilter');

        var sourceType = sourceTypeFilter ? sourceTypeFilter.value : '';
        var dataSource = dataSourceFilter ? dataSourceFilter.value : '';

        container.innerHTML = '<p class="placeholder"><span class="spinner"></span>Loading documents...</p>';

        API.documents.list({
            page: currentPage,
            page_size: pageSize,
            source_type: sourceType || undefined,
            data_source: dataSource || undefined
        }).then(function(data) {
            renderDocuments(data);
        }).catch(function(error) {
            container.innerHTML = '<div class="error-message">Error loading documents: ' +
                API.utils.escapeHtml(error.message) + '</div>';
        });
    }

    /**
     * Get file type icon based on source_type
     * @param {string} sourceType - Source type
     * @returns {string} - Icon HTML
     */
    function getSourceTypeIcon(sourceType) {
        var icons = {
            'webpage': '[WEB]',
            'paper': '[PDF]',
            'upload': '[FILE]'
        };
        return icons[sourceType] || '[DOC]';
    }

    /**
     * Get file extension from file_path
     * @param {string} filePath - File path
     * @returns {string} - File extension or empty string
     */
    function getFileExtension(filePath) {
        if (!filePath) return '';
        var parts = filePath.split('.');
        if (parts.length > 1) {
            return parts[parts.length - 1].toUpperCase();
        }
        return '';
    }

    /**
     * Check if document has downloadable file
     * @param {string} filePath - File path
     * @returns {boolean}
     */
    function isDownloadable(filePath) {
        if (!filePath) return false;
        var ext = filePath.toLowerCase().split('.').pop();
        return ['pdf', 'txt', 'md'].indexOf(ext) !== -1;
    }

    /**
     * Build a readable download label for a file path
     * @param {string} filePath - File path
     * @param {string} fallbackLabel - Fallback label
     * @returns {string}
     */
    function getDownloadLabel(filePath, fallbackLabel) {
        var ext = getFileExtension(filePath).toLowerCase();
        if (ext === 'pdf') return 'PDF';
        if (ext === 'md') return 'Markdown';
        if (ext === 'txt') return 'Text';
        return fallbackLabel || 'Download';
    }

    /**
     * Build action buttons for document download flows
     * @param {Object} doc - Document payload
     * @returns {Array<string>}
     */
    function buildDownloadButtons(doc) {
        var buttons = [
            '<button class="btn small" onclick="DocumentsManager.viewDocument(' + doc.id + ')">View</button>'
        ];

        if (doc.download_url && isDownloadable(doc.file_path)) {
            buttons.push(
                '<a href="' + API.utils.escapeHtml(doc.download_url) + '" class="btn small success" download>' +
                getDownloadLabel(doc.file_path, 'Download') + '</a>'
            );
        }

        if (doc.source_download_url && doc.source_file_path) {
            var sourceLabel = doc.source_type === 'paper'
                ? 'Original PDF'
                : 'Source ' + getDownloadLabel(doc.source_file_path, 'File');
            buttons.push(
                '<a href="' + API.utils.escapeHtml(doc.source_download_url) + '" class="btn small primary" download>' +
                API.utils.escapeHtml(sourceLabel) + '</a>'
            );
        }

        buttons.push(
            '<button class="btn small danger" onclick="DocumentsManager.deleteDocument(' + doc.id + ')">Delete</button>'
        );

        return buttons;
    }

    /**
     * Render documents list
     * @param {Object} data - Documents data from API
     */
    function renderDocuments(data) {
        var container = document.getElementById('docsList');
        if (!container) return;

        if (!data.documents || data.documents.length === 0) {
            container.innerHTML = '<p class="placeholder">No documents found</p>';
            renderPagination(data.total, data.page, data.page_size);
            return;
        }

        var html = data.documents.map(function(doc) {
            var statusClass = 'status-' + doc.status;
            var typeIcon = getSourceTypeIcon(doc.source_type);
            var fileExt = getFileExtension(doc.file_path);

            // Build title with icon
            var titleHtml = '<span class="doc-type-icon">' + typeIcon + '</span> ' +
                API.utils.escapeHtml(doc.title || 'Untitled');

            // Build source info (URL or source description)
            var sourceHtml = '';
            if (doc.url) {
                var displayUrl = API.utils.truncate(doc.url, 50);
                sourceHtml = '<span class="doc-source-link">' +
                    '<a href="' + API.utils.escapeHtml(doc.url) + '" target="_blank" rel="noopener" title="' +
                    API.utils.escapeHtml(doc.url) + '">' + API.utils.escapeHtml(displayUrl) + '</a></span>';
            } else if (doc.source) {
                sourceHtml = '<span class="doc-source-text">' + API.utils.escapeHtml(doc.source) + '</span>';
            }

            // Build meta info line
            var metaItems = [
                '<span class="status-badge ' + statusClass + '">' + doc.status + '</span>',
                '<span class="meta-item"><strong>Type:</strong> ' + API.utils.escapeHtml(doc.source_type) + '</span>',
                '<span class="meta-item"><strong>From:</strong> ' + API.utils.escapeHtml(doc.data_source) + '</span>',
                '<span class="meta-item"><strong>Chunks:</strong> ' + doc.chunk_count + '</span>',
                '<span class="meta-item"><strong>Created:</strong> ' + API.utils.formatRelativeTime(doc.created_at) + '</span>'
            ];

            // Add file type if available
            if (fileExt) {
                metaItems.splice(2, 0, '<span class="meta-item file-type"><strong>Format:</strong> ' + fileExt + '</span>');
            }

            // Build action buttons
            var actionButtons = buildDownloadButtons(doc);

            return '<div class="doc-item">' +
                '<div class="doc-info">' +
                    '<h4>' + titleHtml + '</h4>' +
                    (sourceHtml ? '<div class="doc-source">' + sourceHtml + '</div>' : '') +
                    '<div class="doc-meta">' + metaItems.join('') + '</div>' +
                '</div>' +
                '<div class="doc-actions">' + actionButtons.join('') + '</div>' +
            '</div>';
        }).join('');

        container.innerHTML = html;
        renderPagination(data.total, data.page, data.page_size);
    }

    /**
     * Render pagination controls
     * @param {number} total - Total items
     * @param {number} page - Current page
     * @param {number} size - Page size
     */
    function renderPagination(total, page, size) {
        var container = document.getElementById('docsPagination');
        if (!container) return;

        var totalPages = Math.ceil(total / size);
        if (totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        var html = '';

        // Previous button
        html += '<button ' + (page <= 1 ? 'disabled' : '') +
            ' onclick="DocumentsManager.goToPage(' + (page - 1) + ')">Prev</button>';

        // Page numbers
        var startPage = Math.max(1, page - 2);
        var endPage = Math.min(totalPages, page + 2);

        for (var i = startPage; i <= endPage; i++) {
            html += '<button class="' + (i === page ? 'active' : '') +
                '" onclick="DocumentsManager.goToPage(' + i + ')">' + i + '</button>';
        }

        // Next button
        html += '<button ' + (page >= totalPages ? 'disabled' : '') +
            ' onclick="DocumentsManager.goToPage(' + (page + 1) + ')">Next</button>';

        container.innerHTML = html;
    }

    /**
     * Navigate to a specific page
     * @param {number} page - Page number
     */
    function goToPage(page) {
        currentPage = page;
        loadDocuments();
    }

    /**
     * View document details
     * @param {number} docId - Document ID
     */
    function viewDocument(docId) {
        API.documents.get(docId).then(function(doc) {
            var modal = document.getElementById('docModal');
            var body = document.getElementById('docModalBody');

            if (!modal || !body) return;

            var chunksHtml = '';
            if (doc.chunks && doc.chunks.length > 0) {
                chunksHtml = '<div style="margin-top: 15px"><strong>Chunks (' + doc.chunks.length + '):</strong>';
                doc.chunks.slice(0, 10).forEach(function(chunk, index) {
                    chunksHtml += '<div class="chunk-preview">' +
                        '<strong>Chunk ' + (index + 1) + ' (Level ' + chunk.level + '):</strong><br>' +
                        API.utils.escapeHtml(chunk.content_preview) +
                    '</div>';
                });
                if (doc.chunks.length > 10) {
                    chunksHtml += '<p style="color:#888; margin-top:10px">... and ' +
                        (doc.chunks.length - 10) + ' more chunks</p>';
                }
                chunksHtml += '</div>';
            }

            var html = '<h3>' + API.utils.escapeHtml(doc.title || 'Untitled') + '</h3>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">ID:</span>' +
                    '<span class="detail-value">' + doc.id + '</span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Status:</span>' +
                    '<span class="detail-value"><span class="status-badge status-' + doc.status + '">' +
                        doc.status + '</span></span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Source Type:</span>' +
                    '<span class="detail-value">' + API.utils.escapeHtml(doc.source_type) + '</span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Data Source:</span>' +
                    '<span class="detail-value">' + API.utils.escapeHtml(doc.data_source) + '</span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Chunk Count:</span>' +
                    '<span class="detail-value">' + doc.chunk_count + '</span>' +
                '</div>' +
                (doc.url ? '<div class="detail-row">' +
                    '<span class="detail-label">URL:</span>' +
                    '<span class="detail-value"><a href="' + API.utils.escapeHtml(doc.url) +
                        '" target="_blank" style="color:#4facfe">' +
                        API.utils.truncate(API.utils.escapeHtml(doc.url), 60) + '</a></span>' +
                '</div>' : '') +
                (doc.source ? '<div class="detail-row">' +
                    '<span class="detail-label">Source:</span>' +
                    '<span class="detail-value">' + API.utils.escapeHtml(doc.source) + '</span>' +
                '</div>' : '') +
                (doc.download_url ? '<div class="detail-row">' +
                    '<span class="detail-label">Document File:</span>' +
                    '<span class="detail-value"><a href="' + API.utils.escapeHtml(doc.download_url) + '" download>' +
                        API.utils.escapeHtml(getDownloadLabel(doc.file_path, 'Download')) + '</a></span>' +
                '</div>' : '') +
                (doc.source_download_url ? '<div class="detail-row">' +
                    '<span class="detail-label">Source File:</span>' +
                    '<span class="detail-value"><a href="' + API.utils.escapeHtml(doc.source_download_url) + '" download>' +
                        API.utils.escapeHtml(doc.source_type === 'paper' ? 'Original PDF' : getDownloadLabel(doc.source_file_path, 'Source File')) +
                    '</a></span>' +
                '</div>' : '') +
                '<div class="detail-row">' +
                    '<span class="detail-label">Created:</span>' +
                    '<span class="detail-value">' + API.utils.formatDate(doc.created_at) + '</span>' +
                '</div>' +
                (doc.updated_at ? '<div class="detail-row">' +
                    '<span class="detail-label">Updated:</span>' +
                    '<span class="detail-value">' + API.utils.formatDate(doc.updated_at) + '</span>' +
                '</div>' : '') +
                (doc.metadata ? '<div style="margin-top: 15px"><strong>Metadata:</strong>' +
                    '<pre>' + API.utils.escapeHtml(JSON.stringify(doc.metadata, null, 2)) + '</pre>' +
                '</div>' : '') +
                chunksHtml;

            body.innerHTML = html;
            modal.style.display = 'block';
        }).catch(function(error) {
            alert('Error loading document details: ' + error.message);
        });
    }

    /**
     * Delete a document
     * @param {number} docId - Document ID
     */
    function deleteDocument(docId) {
        if (!confirm('Are you sure you want to delete this document? This will also delete all associated chunks. This action cannot be undone.')) {
            return;
        }

        API.documents.delete(docId).then(function(result) {
            alert(result.message);
            loadDocuments();
        }).catch(function(error) {
            alert('Error deleting document: ' + error.message);
        });
    }

    // Public API
    return {
        init: init,
        loadDocuments: loadDocuments,
        goToPage: goToPage,
        viewDocument: viewDocument,
        deleteDocument: deleteDocument
    };
})();
