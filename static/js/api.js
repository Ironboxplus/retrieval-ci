/**
 * API Module - Encapsulates all API calls for the Retrieval System
 */
var API = (function() {
    'use strict';

    var BASE_URL = '';

    function buildErrorMessage(response, text) {
        var fallback = 'Request failed with status ' + response.status;
        var trimmed = (text || '').trim();
        if (!trimmed) {
            return fallback;
        }

        try {
            var parsed = JSON.parse(trimmed);
            return parsed.detail || parsed.message || parsed.error || fallback;
        } catch (err) {
            if (trimmed.length > 240) {
                trimmed = trimmed.substring(0, 240) + '...';
            }
            return trimmed;
        }
    }

    function parseJsonResponse(response) {
        return response.text().then(function(text) {
            if (!response.ok) {
                throw new Error(buildErrorMessage(response, text));
            }

            if (!text || !text.trim()) {
                return {};
            }

            try {
                return JSON.parse(text);
            } catch (err) {
                throw new Error('Invalid JSON response from server');
            }
        });
    }

    /**
     * Generic request function
     * @param {string} url - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} - Response data
     */
    function request(url, options) {
        options = options || {};
        var defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };

        var mergedOptions = Object.assign({}, defaultOptions, options);

        if (options.headers) {
            mergedOptions.headers = Object.assign({}, defaultOptions.headers, options.headers);
        }

        return window.fetch(BASE_URL + url, mergedOptions).then(parseJsonResponse);
    }

    // ==================== Tasks API ====================

    var tasks = {
        /**
         * Get tasks list
         * @param {Object} params - Query parameters
         * @returns {Promise<Object>}
         */
        list: function(params) {
            params = params || {};
            var queryParts = [];
            if (params.page) queryParts.push('page=' + params.page);
            if (params.page_size) queryParts.push('page_size=' + params.page_size);
            if (params.status) queryParts.push('status=' + params.status);
            if (params.task_type) queryParts.push('task_type=' + params.task_type);

            var query = queryParts.length > 0 ? '?' + queryParts.join('&') : '';
            return request('/api/tasks' + query);
        },

        /**
         * Get single task details
         * @param {number} taskId - Task ID
         * @returns {Promise<Object>}
         */
        get: function(taskId) {
            return request('/api/tasks/' + taskId);
        },

        /**
         * Get workflow graph for a task
         * @param {number} taskId - Task ID
         * @returns {Promise<Object>}
         */
        getWorkflow: function(taskId) {
            return request('/api/tasks/' + taskId + '/workflow');
        },

        /**
         * Get task events
         * @param {number} taskId - Task ID
         * @param {Object} params - Query parameters
         * @returns {Promise<Object>}
         */
        getEvents: function(taskId, params) {
            params = params || {};
            var queryParts = [];
            if (params.after_id) queryParts.push('after_id=' + params.after_id);
            if (params.limit) queryParts.push('limit=' + params.limit);
            var query = queryParts.length > 0 ? '?' + queryParts.join('&') : '';
            return request('/api/tasks/' + taskId + '/events' + query);
        },

        /**
         * Get task resource samples
         * @param {number} taskId - Task ID
         * @param {Object} params - Query parameters
         * @returns {Promise<Object>}
         */
        getResources: function(taskId, params) {
            params = params || {};
            var queryParts = [];
            if (params.after_id) queryParts.push('after_id=' + params.after_id);
            if (params.limit) queryParts.push('limit=' + params.limit);
            var query = queryParts.length > 0 ? '?' + queryParts.join('&') : '';
            return request('/api/tasks/' + taskId + '/resources' + query);
        },

        /**
         * Get task subtasks
         * @param {number} taskId - Task ID
         * @param {Object} params - Query parameters
         * @returns {Promise<Object>}
         */
        getSubtasks: function(taskId, params) {
            params = params || {};
            var queryParts = [];
            if (params.status) queryParts.push('status=' + encodeURIComponent(params.status));
            if (params.limit) queryParts.push('limit=' + params.limit);
            var query = queryParts.length > 0 ? '?' + queryParts.join('&') : '';
            return request('/api/tasks/' + taskId + '/subtasks' + query);
        },

        /**
         * Cancel a task
         * @param {number} taskId - Task ID
         * @returns {Promise<Object>}
         */
        cancel: function(taskId) {
            return request('/api/tasks/' + taskId + '/cancel', { method: 'POST' });
        },

        /**
         * Retry a failed task
         * @param {number} taskId - Task ID
         * @returns {Promise<Object>}
         */
        retry: function(taskId) {
            return request('/api/tasks/' + taskId + '/retry', { method: 'POST' });
        },

        /**
         * Retry a workflow node/subtask
         * @param {number} taskId - Task ID
         * @param {number} subtaskId - Subtask ID
         * @returns {Promise<Object>}
         */
        retrySubtask: function(taskId, subtaskId) {
            return request('/api/tasks/' + taskId + '/subtasks/' + subtaskId + '/retry', {
                method: 'POST'
            });
        },

        /**
         * Apply a workflow node action
         * @param {number} taskId - Task ID
         * @param {number} subtaskId - Node/Subtask ID
         * @param {string} action - Action name
         * @returns {Promise<Object>}
         */
        workflowNodeAction: function(taskId, subtaskId, action) {
            return request('/api/tasks/' + taskId + '/workflow/nodes/' + subtaskId + '/actions/' + action, {
                method: 'POST'
            });
        },

        /**
         * Apply a workflow edge action
         * @param {number} taskId - Task ID
         * @param {number} edgeId - Edge ID
         * @param {string} action - Action name
         * @returns {Promise<Object>}
         */
        workflowEdgeAction: function(taskId, edgeId, action) {
            return request('/api/tasks/' + taskId + '/workflow/edges/' + edgeId + '/actions/' + action, {
                method: 'POST'
            });
        },

        /**
         * Delete a task
         * @param {number} taskId - Task ID
         * @returns {Promise<Object>}
         */
        delete: function(taskId) {
            return request('/api/tasks/' + taskId, { method: 'DELETE' });
        }
    };

    // ==================== Documents API ====================

    var documents = {
        /**
         * Get documents list
         * @param {Object} params - Query parameters
         * @returns {Promise<Object>}
         */
        list: function(params) {
            params = params || {};
            var queryParts = [];
            if (params.page) queryParts.push('page=' + params.page);
            if (params.page_size) queryParts.push('page_size=' + params.page_size);
            if (params.source_type) queryParts.push('source_type=' + params.source_type);
            if (params.data_source) queryParts.push('data_source=' + params.data_source);
            if (params.status) queryParts.push('status=' + params.status);

            var query = queryParts.length > 0 ? '?' + queryParts.join('&') : '';
            return request('/api/documents' + query);
        },

        /**
         * Get single document details
         * @param {number} docId - Document ID
         * @returns {Promise<Object>}
         */
        get: function(docId) {
            return request('/api/documents/' + docId);
        },

        /**
         * Delete a document
         * @param {number} docId - Document ID
         * @returns {Promise<Object>}
         */
        delete: function(docId) {
            return request('/api/documents/' + docId, { method: 'DELETE' });
        },

        /**
         * Upload a document
         * @param {File} file - File to upload
         * @param {string} title - Optional title
         * @param {boolean} autoIndex - Auto index after upload
         * @returns {Promise<Object>}
         */
        upload: function(file, title, autoIndex) {
            var formData = new FormData();
            formData.append('file', file);
            if (title) formData.append('title', title);
            formData.append('auto_index', autoIndex !== false ? 'true' : 'false');

            return window.fetch(BASE_URL + '/api/documents/upload', {
                method: 'POST',
                body: formData
            }).then(parseJsonResponse);
        }
    };

    // ==================== Crawl API ====================

    var crawl = {
        /**
         * Submit URLs for crawling
         * @param {Object} data - Crawl request data
         * @returns {Promise<Object>}
         */
        submitUrls: function(data) {
            return request('/api/crawl/urls', {
                method: 'POST',
                body: JSON.stringify(data)
            });
        },

        /**
         * Upload HTML file to extract links
         * @param {File} file - HTML file
         * @param {Object} options - Upload options
         * @returns {Promise<Object>}
         */
        uploadHtml: function(file, options) {
            options = options || {};
            var formData = new FormData();
            formData.append('file', file);
            formData.append('use_proxy', options.use_proxy ? 'true' : 'false');
            formData.append('auto_index', options.auto_index !== false ? 'true' : 'false');
            formData.append('auto_crawl', options.auto_crawl !== false ? 'true' : 'false');

            return window.fetch(BASE_URL + '/api/crawl/html', {
                method: 'POST',
                body: formData
            }).then(parseJsonResponse);
        },

        /**
         * Get crawl records list
         * @param {Object} params - Query parameters
         * @returns {Promise<Object>}
         */
        getRecords: function(params) {
            params = params || {};
            var queryParts = [];
            if (params.page) queryParts.push('page=' + params.page);
            if (params.page_size) queryParts.push('page_size=' + params.page_size);
            if (params.task_id) queryParts.push('task_id=' + params.task_id);
            if (params.status) queryParts.push('status=' + params.status);

            var query = queryParts.length > 0 ? '?' + queryParts.join('&') : '';
            return request('/api/crawl/records' + query);
        }
    };

    // ==================== Search API ====================

    var search = {
        /**
         * Perform search
         * @param {Object} data - Search request data
         * @returns {Promise<Object>}
         */
        query: function(data) {
            return request('/api/search', {
                method: 'POST',
                body: JSON.stringify(data)
            });
        },

        /**
         * Multi-method search comparison
         * @param {Object} params - Search parameters
         * @returns {Promise<Object>}
         */
        multiMethod: function(params) {
            var queryParts = [];
            queryParts.push('query=' + encodeURIComponent(params.query));
            if (params.top_k) queryParts.push('top_k=' + params.top_k);
            if (params.source_type) queryParts.push('source_type=' + params.source_type);
            if (params.data_source) queryParts.push('data_source=' + params.data_source);

            return request('/api/search/multi-method?' + queryParts.join('&'));
        }
    };

    // ==================== Utility Functions ====================

    var utils = {
        /**
         * Escape HTML to prevent XSS
         * @param {string} text - Text to escape
         * @returns {string}
         */
        escapeHtml: function(text) {
            if (!text) return '';
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        /**
         * Format date for display
         * @param {string} dateStr - ISO date string
         * @returns {string}
         */
        formatDate: function(dateStr) {
            if (!dateStr) return '-';
            var date = new Date(dateStr);
            return date.toLocaleString();
        },

        /**
         * Format relative time
         * @param {string} dateStr - ISO date string
         * @returns {string}
         */
        formatRelativeTime: function(dateStr) {
            if (!dateStr) return '-';
            var date = new Date(dateStr);
            var now = new Date();
            var diff = now - date;

            var seconds = Math.floor(diff / 1000);
            var minutes = Math.floor(seconds / 60);
            var hours = Math.floor(minutes / 60);
            var days = Math.floor(hours / 24);

            if (days > 0) return days + 'd ago';
            if (hours > 0) return hours + 'h ago';
            if (minutes > 0) return minutes + 'm ago';
            return 'just now';
        },

        /**
         * Truncate text with ellipsis
         * @param {string} text - Text to truncate
         * @param {number} maxLength - Maximum length
         * @returns {string}
         */
        truncate: function(text, maxLength) {
            if (!text) return '';
            maxLength = maxLength || 100;
            if (text.length <= maxLength) return text;
            return text.substring(0, maxLength) + '...';
        }
    };

    // Public API
    return {
        tasks: tasks,
        documents: documents,
        crawl: crawl,
        search: search,
        utils: utils
    };
})();
