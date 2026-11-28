/**
 * Tasks Manager Module - Handles task list display and actions
 */
var TasksManager = (function() {
    'use strict';

    var currentPage = 1;
    var pageSize = 20;
    var autoRefreshInterval = null;
    var autoRefreshEnabled = true;

    /**
     * Initialize the tasks manager
     */
    function init() {
        // Bind event listeners
        var refreshBtn = document.getElementById('refreshTasks');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function() {
                loadTasks();
            });
        }

        var statusFilter = document.getElementById('taskStatusFilter');
        if (statusFilter) {
            statusFilter.addEventListener('change', function() {
                currentPage = 1;
                loadTasks();
            });
        }

        var autoRefreshCheckbox = document.getElementById('autoRefreshTasks');
        if (autoRefreshCheckbox) {
            autoRefreshCheckbox.addEventListener('change', function() {
                autoRefreshEnabled = this.checked;
                if (autoRefreshEnabled) {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
            });
        }

        // Initial load
        loadTasks();

        // Start auto refresh
        if (autoRefreshEnabled) {
            startAutoRefresh();
        }
    }

    /**
     * Start auto refresh interval
     */
    function startAutoRefresh() {
        stopAutoRefresh();
        autoRefreshInterval = setInterval(function() {
            var tasksTab = document.getElementById('tasks');
            if (tasksTab && tasksTab.classList.contains('active')) {
                loadTasks();
            }
        }, 5000); // Refresh every 5 seconds
    }

    /**
     * Stop auto refresh interval
     */
    function stopAutoRefresh() {
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
    }

    /**
     * Load tasks from API
     */
    function loadTasks() {
        var container = document.getElementById('tasksList');
        if (!container) return;

        var statusFilter = document.getElementById('taskStatusFilter');
        var status = statusFilter ? statusFilter.value : '';

        // Don't show loading spinner on auto refresh to avoid flicker
        if (!autoRefreshInterval) {
            container.innerHTML = '<p class="placeholder"><span class="spinner"></span>Loading tasks...</p>';
        }

        API.tasks.list({
            page: currentPage,
            page_size: pageSize,
            status: status || undefined
        }).then(function(data) {
            renderTasks(data);
        }).catch(function(error) {
            container.innerHTML = '<div class="error-message">Error loading tasks: ' +
                API.utils.escapeHtml(error.message) + '</div>';
        });
    }

    /**
     * Render tasks list
     * @param {Object} data - Tasks data from API
     */
    function renderTasks(data) {
        var container = document.getElementById('tasksList');
        if (!container) return;

        if (!data.tasks || data.tasks.length === 0) {
            container.innerHTML = '<p class="placeholder">No tasks found</p>';
            renderPagination(data.total, data.page, data.page_size);
            return;
        }

        var html = data.tasks.map(function(task) {
            var progress = task.total > 0 ? Math.round((task.progress / task.total) * 100) : 0;
            var statusClass = 'status-' + task.status;

            return '<div class="task-item ' + statusClass + '">' +
                '<div class="task-info">' +
                    '<h4>' +
                        '<span class="status-badge ' + statusClass + '">' + task.status + '</span>' +
                        ' Task #' + task.id +
                    '</h4>' +
                    '<div class="task-meta">' +
                        '<span>Type: ' + API.utils.escapeHtml(task.task_type) + '</span>' +
                        '<span>Created: ' + API.utils.formatRelativeTime(task.created_at) + '</span>' +
                        (task.error_message ? '<span style="color:#f44336">Error: ' +
                            API.utils.truncate(API.utils.escapeHtml(task.error_message), 50) + '</span>' : '') +
                    '</div>' +
                '</div>' +
                '<div class="task-progress">' +
                    '<div class="progress-bar">' +
                        '<div class="progress-bar-fill" style="width: ' + progress + '%"></div>' +
                    '</div>' +
                    '<div class="progress-text">' + task.progress + ' / ' + task.total + ' (' + progress + '%)</div>' +
                '</div>' +
                '<div class="task-actions">' +
                    '<button class="btn small" onclick="TasksManager.viewTask(' + task.id + ')">View</button>' +
                    '<button class="btn small primary" onclick="TasksManager.openWorkflow(' + task.id + ')">Workflow</button>' +
                    (canCancel(task) ?
                        '<button class="btn small warning" onclick="TasksManager.cancelTask(' + task.id + ')">Cancel</button>' : '') +
                    (canRetry(task) ?
                        '<button class="btn small success" onclick="TasksManager.retryTask(' + task.id + ')">Retry</button>' : '') +
                    (canDelete(task) ?
                        '<button class="btn small danger" onclick="TasksManager.deleteTask(' + task.id + ')">Delete</button>' : '') +
                '</div>' +
            '</div>';
        }).join('');

        container.innerHTML = html;
        renderPagination(data.total, data.page, data.page_size);
    }

    /**
     * Check if task can be cancelled
     * @param {Object} task - Task object
     * @returns {boolean}
     */
    function canCancel(task) {
        return task.status === 'pending' || task.status === 'running';
    }

    /**
     * Check if task can be retried
     * @param {Object} task - Task object
     * @returns {boolean}
     */
    function canRetry(task) {
        return task.status === 'failed' && task.retry_count < task.max_retries;
    }

    /**
     * Check if task can be deleted
     * @param {Object} task - Task object
     * @returns {boolean}
     */
    function canDelete(task) {
        return task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled';
    }

    /**
     * Render pagination controls
     * @param {number} total - Total items
     * @param {number} page - Current page
     * @param {number} size - Page size
     */
    function renderPagination(total, page, size) {
        var container = document.getElementById('tasksPagination');
        if (!container) return;

        var totalPages = Math.ceil(total / size);
        if (totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        var html = '';

        // Previous button
        html += '<button ' + (page <= 1 ? 'disabled' : '') +
            ' onclick="TasksManager.goToPage(' + (page - 1) + ')">Prev</button>';

        // Page numbers
        var startPage = Math.max(1, page - 2);
        var endPage = Math.min(totalPages, page + 2);

        for (var i = startPage; i <= endPage; i++) {
            html += '<button class="' + (i === page ? 'active' : '') +
                '" onclick="TasksManager.goToPage(' + i + ')">' + i + '</button>';
        }

        // Next button
        html += '<button ' + (page >= totalPages ? 'disabled' : '') +
            ' onclick="TasksManager.goToPage(' + (page + 1) + ')">Next</button>';

        container.innerHTML = html;
    }

    /**
     * Navigate to a specific page
     * @param {number} page - Page number
     */
    function goToPage(page) {
        currentPage = page;
        loadTasks();
    }

    /**
     * View task details
     * @param {number} taskId - Task ID
     */
    function viewTask(taskId) {
        API.tasks.get(taskId).then(function(task) {
            var modal = document.getElementById('taskModal');
            var body = document.getElementById('taskModalBody');

            if (!modal || !body) return;

            var html = '<h3>Task #' + task.id + ' Details</h3>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Status:</span>' +
                    '<span class="detail-value"><span class="status-badge status-' + task.status + '">' +
                        task.status + '</span></span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Type:</span>' +
                    '<span class="detail-value">' + API.utils.escapeHtml(task.task_type) + '</span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Progress:</span>' +
                    '<span class="detail-value">' + task.progress + ' / ' + task.total + '</span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Priority:</span>' +
                    '<span class="detail-value">' + task.priority + '</span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Retry Count:</span>' +
                    '<span class="detail-value">' + task.retry_count + ' / ' + task.max_retries + '</span>' +
                '</div>' +
                '<div class="detail-row">' +
                    '<span class="detail-label">Created:</span>' +
                    '<span class="detail-value">' + API.utils.formatDate(task.created_at) + '</span>' +
                '</div>' +
                (task.duration_seconds ? '<div class="detail-row">' +
                    '<span class="detail-label">Duration:</span>' +
                    '<span class="detail-value">' + task.duration_seconds + 's</span>' +
                '</div>' : '') +
                (task.started_at ? '<div class="detail-row">' +
                    '<span class="detail-label">Started:</span>' +
                    '<span class="detail-value">' + API.utils.formatDate(task.started_at) + '</span>' +
                '</div>' : '') +
                (task.completed_at ? '<div class="detail-row">' +
                    '<span class="detail-label">Completed:</span>' +
                    '<span class="detail-value">' + API.utils.formatDate(task.completed_at) + '</span>' +
                '</div>' : '') +
                (task.error_message ? '<div class="detail-row">' +
                    '<span class="detail-label">Error:</span>' +
                    '<span class="detail-value" style="color:#f44336">' +
                        API.utils.escapeHtml(task.error_message) + '</span>' +
                '</div>' : '') +
                '<div style="margin-top: 16px; display:flex; gap:10px; flex-wrap:wrap;">' +
                    '<button class="btn primary" onclick="TasksManager.openWorkflow(' + task.id + ')">Open Workflow Graph</button>' +
                '</div>' +
                (task.params ? '<div style="margin-top: 15px"><strong>Parameters:</strong>' +
                    '<pre>' + API.utils.escapeHtml(JSON.stringify(task.params, null, 2)) + '</pre>' +
                '</div>' : '') +
                (task.result ? '<div style="margin-top: 15px"><strong>Result:</strong>' +
                    '<pre>' + API.utils.escapeHtml(JSON.stringify(task.result, null, 2)) + '</pre>' +
                '</div>' : '');

            body.innerHTML = html;
            modal.style.display = 'block';
        }).catch(function(error) {
            alert('Error loading task details: ' + error.message);
        });
    }

    /**
     * Open the workflow graph page for a task
     * @param {number} taskId - Task ID
     */
    function openWorkflow(taskId) {
        window.open('/static/task-workflow.html?task_id=' + encodeURIComponent(taskId), '_blank');
    }

    /**
     * Cancel a task
     * @param {number} taskId - Task ID
     */
    function cancelTask(taskId) {
        if (!confirm('Are you sure you want to cancel this task?')) return;

        API.tasks.cancel(taskId).then(function(result) {
            alert(result.message);
            loadTasks();
        }).catch(function(error) {
            alert('Error cancelling task: ' + error.message);
        });
    }

    /**
     * Retry a failed task
     * @param {number} taskId - Task ID
     */
    function retryTask(taskId) {
        if (!confirm('Are you sure you want to retry this task?')) return;

        API.tasks.retry(taskId).then(function(result) {
            alert(result.message);
            loadTasks();
        }).catch(function(error) {
            alert('Error retrying task: ' + error.message);
        });
    }

    /**
     * Delete a task
     * @param {number} taskId - Task ID
     */
    function deleteTask(taskId) {
        if (!confirm('Are you sure you want to delete this task? This action cannot be undone.')) return;

        API.tasks.delete(taskId).then(function(result) {
            alert(result.message);
            loadTasks();
        }).catch(function(error) {
            alert('Error deleting task: ' + error.message);
        });
    }

    // Public API
    return {
        init: init,
        loadTasks: loadTasks,
        goToPage: goToPage,
        viewTask: viewTask,
        openWorkflow: openWorkflow,
        cancelTask: cancelTask,
        retryTask: retryTask,
        deleteTask: deleteTask
    };
})();
