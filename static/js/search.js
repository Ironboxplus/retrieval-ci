/**
 * Search Manager Module - Handles search functionality
 */
var SearchManager = (function() {
    'use strict';

    /**
     * Initialize the search manager
     */
    function init() {
        // Bind search button
        var searchBtn = document.getElementById('searchBtn');
        if (searchBtn) {
            searchBtn.addEventListener('click', performSearch);
        }

        // Bind enter key on search input
        var searchInput = document.getElementById('searchQuery');
        if (searchInput) {
            searchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    performSearch();
                }
            });
        }
    }

    /**
     * Perform search
     */
    function performSearch() {
        var queryEl = document.getElementById('searchQuery');
        var methodEl = document.getElementById('searchMethod');
        var topKEl = document.getElementById('topK');
        var aggregateEl = document.getElementById('aggregateByDoc');
        var sourceTypeEl = document.getElementById('searchSourceType');
        var dataSourceEl = document.getElementById('searchDataSource');
        var resultsContainer = document.getElementById('searchResults');

        if (!queryEl || !resultsContainer) return;

        var query = queryEl.value.trim();
        if (!query) {
            resultsContainer.innerHTML = '<p class="placeholder">Please enter a search query</p>';
            return;
        }

        resultsContainer.innerHTML = '<p class="placeholder"><span class="spinner"></span>Searching...</p>';

        var searchData = {
            query: query,
            method: methodEl ? methodEl.value : 'enhanced',
            top_k: topKEl ? parseInt(topKEl.value, 10) || 20 : 20,
            aggregate_by_doc: aggregateEl ? aggregateEl.checked : true
        };

        // Add optional filters
        if (sourceTypeEl && sourceTypeEl.value) {
            searchData.source_type = sourceTypeEl.value;
        }
        if (dataSourceEl && dataSourceEl.value) {
            searchData.data_source = dataSourceEl.value;
        }

        API.search.query(searchData).then(function(data) {
            renderResults(data);
        }).catch(function(error) {
            resultsContainer.innerHTML = '<div class="error-message">Search error: ' +
                API.utils.escapeHtml(error.message) + '</div>';
        });
    }

    /**
     * Render search results
     * @param {Object} data - Search results from API
     */
    function renderResults(data) {
        var container = document.getElementById('searchResults');
        if (!container) return;

        // Check for aggregated results first
        if (data.aggregated && data.aggregated.length > 0) {
            renderAggregatedResults(container, data);
            return;
        }

        // Render flat results
        if (data.results && data.results.length > 0) {
            renderFlatResults(container, data);
            return;
        }

        container.innerHTML = '<p class="placeholder">No results found for your query</p>';
    }

    /**
     * Render aggregated search results
     * @param {HTMLElement} container - Container element
     * @param {Object} data - Search results data
     */
    function renderAggregatedResults(container, data) {
        var html = '<h3>Found ' + data.total_results + ' results in ' +
            data.aggregated.length + ' documents</h3>';

        html += data.aggregated.map(function(agg) {
            var chunksHtml = '';
            if (agg.chunks && agg.chunks.length > 0) {
                chunksHtml = '<div class="chunks-container">';
                agg.chunks.slice(0, 3).forEach(function(chunk) {
                    chunksHtml += '<div class="chunk-preview">' +
                        '<p class="content">' +
                            API.utils.escapeHtml(API.utils.truncate(chunk.content, 300)) +
                        '</p>' +
                        '<span class="score">Score: ' + (chunk.score != null ? chunk.score.toFixed(4) : 'N/A') + '</span>' +
                    '</div>';
                });
                if (agg.chunks.length > 3) {
                    chunksHtml += '<p style="color:#888; font-size:0.85rem">... and ' +
                        (agg.chunks.length - 3) + ' more matching chunks</p>';
                }
                chunksHtml += '</div>';
            }

            return '<div class="result-item aggregated-result">' +
                '<h4>' +
                    API.utils.escapeHtml(agg.doc_title || 'Untitled') +
                    '<span class="hit-count">' + agg.hit_count + ' hits</span>' +
                '</h4>' +
                '<p class="source">' + API.utils.escapeHtml(agg.source || '') + '</p>' +
                '<div class="score">' +
                    'Total Score: ' + (agg.total_score != null ? agg.total_score.toFixed(4) : 'N/A') +
                    ' | Avg Score: ' + (agg.avg_score != null ? agg.avg_score.toFixed(4) : 'N/A') +
                    ' | Max Score: ' + (agg.max_score != null ? agg.max_score.toFixed(4) : 'N/A') +
                '</div>' +
                chunksHtml +
            '</div>';
        }).join('');

        container.innerHTML = html;
    }

    /**
     * Render flat search results
     * @param {HTMLElement} container - Container element
     * @param {Object} data - Search results data
     */
    function renderFlatResults(container, data) {
        var html = '<h3>Found ' + data.total_results + ' results</h3>';

        html += data.results.map(function(result) {
            var methodInfo = '';
            if (result.metadata) {
                if (result.metadata.method) {
                    methodInfo = ' | Method: ' + result.metadata.method;
                }
                if (result.metadata.bm25_score != null) {
                    methodInfo += ' | BM25: ' + result.metadata.bm25_score.toFixed(4);
                }
                if (result.metadata.vector_score != null) {
                    methodInfo += ' | Vector: ' + result.metadata.vector_score.toFixed(4);
                }
            }

            return '<div class="result-item">' +
                '<h4>' + API.utils.escapeHtml(result.doc_title || 'Untitled') + '</h4>' +
                '<p class="source">' + API.utils.escapeHtml(result.source || '') + '</p>' +
                '<p class="content">' +
                    API.utils.escapeHtml(API.utils.truncate(result.content, 500)) +
                '</p>' +
                '<div class="score">' +
                    'Score: ' + (result.score != null ? result.score.toFixed(4) : 'N/A') +
                    methodInfo +
                    (result.level !== undefined ? ' | Level: ' + result.level : '') +
                '</div>' +
            '</div>';
        }).join('');

        container.innerHTML = html;
    }

    // Public API
    return {
        init: init,
        performSearch: performSearch
    };
})();
