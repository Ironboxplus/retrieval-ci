/* global API, cytoscape, dagre, EventSource */
(function() {
    'use strict';

    var dagrePlugin = window.cytoscapeDagre || window.dagreCytoscape || window.cytoscapeDagrePlugin;
    if (window.cytoscape && dagrePlugin) {
        window.cytoscape.use(dagrePlugin);
    }

    var taskId = null;
    var cy = null;
    var currentTask = null;
    var currentWorkflow = null;
    var eventSource = null;
    var refreshTimer = null;
    var eventsBuffer = [];

    function workflowLayoutOptions() {
        return {
            name: 'dagre',
            rankDir: 'LR',
            nodeSep: 54,
            rankSep: 90,
            edgeSep: 24,
            animate: true,
            animationDuration: 220
        };
    }

    function isTerminalStatus(status) {
        status = (status || '').toLowerCase();
        return status === 'completed' || status === 'failed' || status === 'cancelled' || status === 'timeout';
    }

    function stopStream() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    function init() {
        var params = new URLSearchParams(window.location.search);
        taskId = params.get('task_id');
        if (!taskId) {
            renderError('Missing task_id in query string.');
            return;
        }

        var refreshBtn = document.getElementById('workflowRefreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', loadAll);
        }

        var fitBtn = document.getElementById('workflowFitBtn');
        if (fitBtn) {
            fitBtn.addEventListener('click', function() {
                if (cy) cy.fit(undefined, 42);
            });
        }

        loadAll();
        connectStream();
    }

    function renderError(message) {
        var graphEl = document.getElementById('workflowGraph');
        if (graphEl) {
            graphEl.innerHTML = '<div class="workflow-error">' + API.utils.escapeHtml(message) + '</div>';
        }
    }

    function loadAll() {
        Promise.all([
            API.tasks.get(taskId),
            API.tasks.getWorkflow(taskId),
            API.tasks.getEvents(taskId, { limit: 40 })
        ]).then(function(results) {
            currentTask = results[0];
            currentWorkflow = results[1];
            eventsBuffer = (results[2] && results[2].events) ? results[2].events.slice(-40) : [];
            renderTaskSummary(currentTask, currentWorkflow);
            renderRuntime(currentTask);
            renderGraph(currentWorkflow);
            renderEvents(eventsBuffer);
        }).catch(function(error) {
            renderError(error.message || 'Failed to load workflow.');
        });
    }

    function connectStream() {
        stopStream();

        setLivePill('connecting', 'connecting');
        eventSource = new EventSource('/api/tasks/' + encodeURIComponent(taskId) + '/workflow/stream');
        eventSource.addEventListener('open', function() {
            setLivePill('active', 'live');
        });
        eventSource.addEventListener('error', function() {
            if (currentTask && isTerminalStatus(currentTask.status)) {
                setLivePill('completed', currentTask.status || 'completed');
                stopStream();
                return;
            }
            setLivePill('failed', 'reconnecting');
        });
        eventSource.addEventListener('snapshot', function(evt) {
            try {
                var payload = JSON.parse(evt.data);
                currentTask = payload.task || currentTask;
                currentWorkflow = payload.workflow || currentWorkflow;
                renderRuntime(currentTask);
                renderTaskSummary(currentTask, currentWorkflow);
                renderGraph(currentWorkflow);
                if (currentTask && isTerminalStatus(currentTask.status)) {
                    setLivePill('completed', currentTask.status || 'completed');
                    stopStream();
                }
            } catch (err) {}
        });
        eventSource.addEventListener('workflow', function(evt) {
            try {
                currentWorkflow = JSON.parse(evt.data);
                renderGraph(currentWorkflow);
                renderTaskSummary(currentTask, currentWorkflow);
            } catch (err) {}
        });
        eventSource.addEventListener('task-event', function(evt) {
            try {
                var payload = JSON.parse(evt.data);
                pushEvent(payload);
            } catch (err) {}
            scheduleRefresh();
        });
        eventSource.addEventListener('resource', function() {
            scheduleRefresh();
        });
        eventSource.addEventListener('final', function(evt) {
            try {
                var payload = JSON.parse(evt.data);
                currentTask = payload.task || currentTask;
                currentWorkflow = payload.workflow || currentWorkflow;
                renderRuntime(currentTask);
                renderTaskSummary(currentTask, currentWorkflow);
                renderGraph(currentWorkflow);
            } catch (err) {}
            setLivePill('completed', 'completed');
            stopStream();
        });
    }

    function scheduleRefresh() {
        if (refreshTimer) {
            window.clearTimeout(refreshTimer);
        }
        refreshTimer = window.setTimeout(function() {
            API.tasks.get(taskId).then(function(task) {
                currentTask = task;
                renderRuntime(currentTask);
                renderTaskSummary(currentTask, currentWorkflow);
                return API.tasks.getWorkflow(taskId);
            }).then(function(workflow) {
                currentWorkflow = workflow;
                renderGraph(currentWorkflow);
                renderTaskSummary(currentTask, currentWorkflow);
            }).catch(function() {});
        }, 350);
    }

    function setLivePill(status, label) {
        var pill = document.getElementById('workflowLivePill');
        if (!pill) return;
        pill.className = 'pill ' + status;
        pill.innerHTML = '<span class="status-dot"></span><span>' + API.utils.escapeHtml(label) + '</span>';
    }

    function setStatusPill(status) {
        var pill = document.getElementById('workflowStatusPill');
        if (!pill) return;
        pill.className = 'pill ' + status;
        pill.innerHTML = '<span class="status-dot"></span><span>' + API.utils.escapeHtml(status) + '</span>';
    }

    function renderTaskSummary(task, workflow) {
        if (!task) return;
        var title = document.getElementById('workflowTitle');
        var subtitle = document.getElementById('workflowSubtitle');
        var summary = document.getElementById('workflowSummary');
        if (title) {
            title.textContent = 'Task #' + task.id + ' · ' + task.task_type;
        }
        if (subtitle) {
            var nodes = workflow && workflow.summary ? workflow.summary.node_count || 0 : 0;
            var edges = workflow && workflow.summary ? workflow.summary.edge_count || 0 : 0;
            subtitle.textContent = '实时工作流图，包含 ' + nodes + ' 个节点和 ' + edges + ' 条边。节点与边都支持点击查看及细粒度操作。';
        }
        setStatusPill(task.status || 'unknown');

        if (!summary) return;
        var wfSummary = workflow && workflow.summary ? workflow.summary : {};
        var resource = task.resource_summary || {};
        var peakRssGb = resource.peak_rss_bytes ? (resource.peak_rss_bytes / 1024 / 1024 / 1024).toFixed(2) : '0.00';
        summary.innerHTML =
            summaryCard('Task Status', task.status || 'unknown', 'Worker: ' + (task.worker_id || '-')) +
            summaryCard('Graph Nodes', String(wfSummary.node_count || 0), 'Edges: ' + String(wfSummary.edge_count || 0)) +
            summaryCard('Running Nodes', String(wfSummary.running_nodes || 0), 'Failed: ' + String(wfSummary.failed_nodes || 0)) +
            summaryCard('Peak RSS', peakRssGb + ' GB', 'Samples: ' + String(resource.sample_count || 0));
    }

    function summaryCard(label, value, detail) {
        return '<div class="summary-card">' +
            '<div class="summary-label">' + API.utils.escapeHtml(label) + '</div>' +
            '<div class="summary-value">' + API.utils.escapeHtml(value) + '</div>' +
            '<div class="summary-detail">' + API.utils.escapeHtml(detail) + '</div>' +
        '</div>';
    }

    function renderRuntime(task) {
        var panel = document.getElementById('runtimePanel');
        if (!panel || !task) return;
        var health = task.health || {};
        var resource = task.resource_summary || {};
        var latest = resource.latest || {};
        panel.innerHTML =
            kv('Task', '#' + task.id + ' · ' + (task.task_type || '-')) +
            kv('Status', task.status || '-') +
            kv('Duration', task.duration_seconds ? task.duration_seconds + 's' : '-') +
            kv('Heartbeat', health.heartbeat_age_seconds != null ? health.heartbeat_age_seconds + 's' : '-') +
            kv('Health', health.is_healthy ? 'healthy' : 'attention') +
            kv('Peak RSS', resource.peak_rss_bytes ? Math.round(resource.peak_rss_bytes / 1024 / 1024) + ' MB' : '-') +
            kv('Latest CPU', latest.cpu_percent != null ? latest.cpu_percent + '%' : '-') +
            kv('Latest Threads', latest.thread_count != null ? String(latest.thread_count) : '-');
    }

    function kv(label, value) {
        return '<strong>' + API.utils.escapeHtml(label) + '</strong><span>' + API.utils.escapeHtml(String(value)) + '</span>';
    }

    function renderGraph(workflow) {
        if (!workflow || !workflow.nodes) return;

        var elements = [];
        workflow.nodes.forEach(function(node) {
            elements.push({
                data: {
                    id: String(node.key),
                    label: node.name || node.key,
                    status: node.status,
                    type: node.type,
                    raw: node
                }
            });
        });
        (workflow.edges || []).forEach(function(edge) {
            elements.push({
                data: {
                    id: 'edge:' + String(edge.key),
                    source: String(edge.source),
                    target: String(edge.target),
                    label: edge.label || '',
                    status: edge.status,
                    raw: edge
                }
            });
        });

        if (!cy) {
            cy = cytoscape({
                container: document.getElementById('workflowGraph'),
                elements: elements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'shape': 'round-rectangle',
                            'width': 184,
                            'height': 74,
                            'padding': '10px',
                            'background-color': function(ele) { return nodeColor(ele.data('status')); },
                            'border-width': 2,
                            'border-color': '#ffffff',
                            'label': function(ele) { return ele.data('label') + '\n' + ele.data('status'); },
                            'text-wrap': 'wrap',
                            'text-max-width': 140,
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': 12,
                            'font-weight': 600,
                            'color': '#102033',
                            'shadow-blur': 18,
                            'shadow-color': 'rgba(16,32,51,0.18)',
                            'shadow-offset-x': 0,
                            'shadow-offset-y': 8
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'curve-style': 'bezier',
                            'width': 3,
                            'line-color': function(ele) { return edgeColor(ele.data('status')); },
                            'target-arrow-color': function(ele) { return edgeColor(ele.data('status')); },
                            'target-arrow-shape': 'triangle',
                            'label': 'data(label)',
                            'font-size': 10,
                            'text-background-color': 'rgba(255,255,255,0.92)',
                            'text-background-opacity': 1,
                            'text-background-padding': 4,
                            'text-rotation': 'autorotate',
                            'color': '#435066'
                        }
                    },
                    {
                        selector: ':selected',
                        style: {
                            'border-color': '#0b6bcb',
                            'border-width': 4,
                            'line-color': '#0b6bcb',
                            'target-arrow-color': '#0b6bcb'
                        }
                    }
                ],
                layout: workflowLayoutOptions()
            });

            cy.on('tap', 'node', function(evt) {
                showSelection('node', evt.target.data('raw'));
            });
            cy.on('tap', 'edge', function(evt) {
                showSelection('edge', evt.target.data('raw'));
            });
            cy.on('tap', function(evt) {
                if (evt.target === cy) {
                    showSelection(null, null);
                }
            });
        } else {
            cy.elements().remove();
            cy.add(elements);
            cy.layout(workflowLayoutOptions()).run();
        }

        cy.fit(undefined, 42);
    }

    function fitGraph() {
        if (!cy) return false;
        cy.fit(undefined, 42);
        return true;
    }

    function panBy(deltaX, deltaY) {
        if (!cy) return false;
        var pan = cy.pan();
        cy.pan({
            x: pan.x + Number(deltaX || 0),
            y: pan.y + Number(deltaY || 0)
        });
        return true;
    }

    function zoomTo(level) {
        if (!cy) return false;
        var numericLevel = Number(level);
        if (!isFinite(numericLevel) || numericLevel <= 0) {
            return false;
        }
        cy.zoom(numericLevel);
        return true;
    }

    function rerunLayout() {
        if (!cy) return false;
        cy.layout(workflowLayoutOptions()).run();
        return true;
    }

    function findNodeByRawId(subtaskId) {
        if (!cy) return null;
        return cy.nodes().toArray().find(function(node) {
            var raw = node.data('raw') || {};
            return String(raw.id) === String(subtaskId);
        }) || null;
    }

    function findEdgeByRawId(edgeId) {
        if (!cy) return null;
        return cy.edges().toArray().find(function(edge) {
            var raw = edge.data('raw') || {};
            return String(raw.id) === String(edgeId);
        }) || null;
    }

    function selectNode(subtaskId) {
        var node = findNodeByRawId(subtaskId);
        if (!node) return false;
        cy.elements().unselect();
        node.select();
        node.emit('tap');
        return true;
    }

    function selectEdge(edgeId) {
        var edge = findEdgeByRawId(edgeId);
        if (!edge) return false;
        cy.elements().unselect();
        edge.select();
        edge.emit('tap');
        return true;
    }

    function graphNodeSnapshot(node) {
        var raw = node.data('raw') || {};
        var position = node.position();
        var renderedPosition = node.renderedPosition();
        return {
            id: node.id(),
            raw_id: raw.id,
            label: node.data('label'),
            status: node.data('status'),
            position: { x: position.x, y: position.y },
            rendered_position: { x: renderedPosition.x, y: renderedPosition.y }
        };
    }

    function graphEdgeSnapshot(edge) {
        var raw = edge.data('raw') || {};
        var renderedMidpoint = typeof edge.renderedMidpoint === 'function' ? edge.renderedMidpoint() : null;
        return {
            id: edge.id(),
            raw_id: raw.id,
            source: edge.data('source'),
            target: edge.data('target'),
            status: edge.data('status'),
            rendered_midpoint: renderedMidpoint ? { x: renderedMidpoint.x, y: renderedMidpoint.y } : null
        };
    }

    function getGraphState() {
        if (!cy) {
            return {
                ready: false,
                node_count: 0,
                edge_count: 0,
                zoom: 0,
                pan: { x: 0, y: 0 },
                nodes: [],
                edges: [],
                selected_ids: [],
                user_panning_enabled: false,
                user_zooming_enabled: false
            };
        }

        return {
            ready: true,
            node_count: cy.nodes().length,
            edge_count: cy.edges().length,
            zoom: cy.zoom(),
            pan: cy.pan(),
            nodes: cy.nodes().toArray().map(graphNodeSnapshot),
            edges: cy.edges().toArray().map(graphEdgeSnapshot),
            selected_ids: cy.$(':selected').toArray().map(function(ele) { return ele.id(); }),
            user_panning_enabled: cy.userPanningEnabled(),
            user_zooming_enabled: cy.userZoomingEnabled()
        };
    }

    function nodeColor(status) {
        switch ((status || '').toLowerCase()) {
            case 'running': return '#dff1ff';
            case 'completed': return '#dcf6e5';
            case 'failed': return '#fde5e2';
            case 'cancelled': return '#f3eadc';
            case 'skipped': return '#f4efe9';
            default: return '#eef2f7';
        }
    }

    function edgeColor(status) {
        switch ((status || '').toLowerCase()) {
            case 'disabled': return '#b42318';
            case 'blocked': return '#b42318';
            case 'traversed': return '#18794e';
            default: return '#0b6bcb';
        }
    }

    function showSelection(kind, payload) {
        var panel = document.getElementById('selectionPanel');
        if (!panel) return;
        if (!kind || !payload) {
            panel.innerHTML = '点击节点或边查看详情和可执行动作。';
            return;
        }

        if (kind === 'node') {
            panel.innerHTML = renderSelectionDetails(payload, 'Node') +
                renderActionButtons(payload.allowed_actions || [], function(action) {
                    return 'window.TaskWorkflowPage.runNodeAction(' + payload.id + ', "' + action + '")';
                });
            return;
        }

        panel.innerHTML = renderSelectionDetails(payload, 'Edge') +
            renderActionButtons(payload.allowed_actions || [], function(action) {
                return 'window.TaskWorkflowPage.runEdgeAction(' + payload.id + ', "' + action + '")';
            });
    }

    function renderSelectionDetails(payload, label) {
        var detailBlocks = '<div style="margin-top:14px"><strong>Metadata</strong><pre>' +
            API.utils.escapeHtml(JSON.stringify(payload.metadata || {}, null, 2)) +
            '</pre></div>';
        if (payload.result) {
            detailBlocks += '<div style="margin-top:14px"><strong>Result</strong><pre>' +
                API.utils.escapeHtml(JSON.stringify(payload.result, null, 2)) +
                '</pre></div>';
        }
        return '<div class="detail-kv">' +
            kv(label, payload.name || payload.key || payload.id) +
            kv('Status', payload.status || '-') +
            kv('Type', payload.type || payload.label || '-') +
            kv('Stage', payload.stage || '-') +
            kv('Retries', payload.retry_count != null ? payload.retry_count + ' / ' + (payload.max_retries || '-') : '-') +
            kv('Message', payload.message || payload.error_message || '-') +
        '</div>' +
        detailBlocks;
    }

    function renderActionButtons(actions, actionHrefBuilder) {
        if (!actions || !actions.length) {
            return '<div class="selection-empty" style="margin-top:12px">This element has no interactive actions.</div>';
        }

        return '<div class="action-buttons">' + actions.filter(function(action) {
            return action !== 'inspect';
        }).map(function(action) {
            var btnClass = action === 'retry' || action === 'enable' ? 'btn primary' : 'btn warning';
            return '<button class="' + btnClass + '" onclick="' + actionHrefBuilder(action) + '">' +
                API.utils.escapeHtml(action) + '</button>';
        }).join('') + '</div>';
    }

    function renderEvents(events) {
        var stream = document.getElementById('eventStream');
        if (!stream) return;
        if (!events || !events.length) {
            stream.innerHTML = '<div class="selection-empty">Waiting for live events...</div>';
            return;
        }

        stream.innerHTML = events.slice(-40).reverse().map(function(evt) {
            return '<div class="event-item">' +
                '<div class="event-meta">' +
                    '<span>' + API.utils.escapeHtml(evt.event_type || 'event') + '</span>' +
                    '<span>' + API.utils.escapeHtml(API.utils.formatDate(evt.created_at)) + '</span>' +
                '</div>' +
                '<div>' + API.utils.escapeHtml(evt.message || '-') + '</div>' +
            '</div>';
        }).join('');
    }

    function pushEvent(evt) {
        eventsBuffer.push(evt);
        if (eventsBuffer.length > 80) {
            eventsBuffer = eventsBuffer.slice(-80);
        }
        renderEvents(eventsBuffer);
    }

    function runNodeAction(subtaskId, action) {
        API.tasks.workflowNodeAction(taskId, subtaskId, action).then(function(resp) {
            window.alert(resp.message);
            loadAll();
        }).catch(function(error) {
            window.alert('Node action failed: ' + error.message);
        });
    }

    function runEdgeAction(edgeId, action) {
        API.tasks.workflowEdgeAction(taskId, edgeId, action).then(function(resp) {
            window.alert(resp.message);
            loadAll();
        }).catch(function(error) {
            window.alert('Edge action failed: ' + error.message);
        });
    }

    window.TaskWorkflowPage = {
        runNodeAction: runNodeAction,
        runEdgeAction: runEdgeAction,
        fitGraph: fitGraph,
        panBy: panBy,
        zoomTo: zoomTo,
        rerunLayout: rerunLayout,
        selectNode: selectNode,
        selectEdge: selectEdge,
        getGraphState: getGraphState
    };

    document.addEventListener('DOMContentLoaded', init);
})();
