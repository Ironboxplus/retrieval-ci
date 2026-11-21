// Retrieval System Frontend

const API_BASE = '';

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');

        // Load data for specific tabs
        if (tab.dataset.tab === 'documents') loadDocuments();
        if (tab.dataset.tab === 'settings') loadStats();
    });
});

// Search
document.getElementById('searchBtn').addEventListener('click', performSearch);
document.getElementById('searchQuery').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

async function performSearch() {
    const query = document.getElementById('searchQuery').value.trim();
    if (!query) return;

    const method = document.getElementById('searchMethod').value;
    const topK = parseInt(document.getElementById('topK').value);
    const aggregateByDoc = document.getElementById('aggregateByDoc').checked;

    const resultsContainer = document.getElementById('searchResults');
    resultsContainer.innerHTML = '<p class="status-loading"><span class="spinner"></span>Searching...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                method,
                top_k: topK,
                aggregate_by_doc: aggregateByDoc
            })
        });

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        resultsContainer.innerHTML = `<p class="status-error">Error: ${error.message}</p>`;
    }
}

function displayResults(data) {
    const container = document.getElementById('searchResults');

    if (data.aggregated && data.aggregated.length > 0) {
        // Display aggregated results
        container.innerHTML = `
            <h3>Found ${data.total_results} results in ${data.aggregated.length} documents</h3>
            ${data.aggregated.map(agg => `
                <div class="result-item aggregated-result">
                    <h4>${escapeHtml(agg.doc_title)}
                        <span class="hit-count">${agg.hit_count} hits</span>
                    </h4>
                    <p class="source">${escapeHtml(agg.source)}</p>
                    <div class="score">Total Score: ${agg.total_score.toFixed(4)} | Avg: ${agg.avg_score.toFixed(4)}</div>
                    <div class="chunks">
                        ${agg.chunks.slice(0, 3).map(chunk => `
                            <div class="chunk-preview">
                                <p class="content">${escapeHtml(chunk.content.substring(0, 300))}...</p>
                                <span class="score">Score: ${chunk.score.toFixed(4)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
        `;
    } else if (data.results && data.results.length > 0) {
        // Display flat results
        container.innerHTML = `
            <h3>Found ${data.total_results} results</h3>
            ${data.results.map(result => `
                <div class="result-item">
                    <h4>${escapeHtml(result.doc_title)}</h4>
                    <p class="source">${escapeHtml(result.source)}</p>
                    <p class="content">${escapeHtml(result.content.substring(0, 500))}...</p>
                    <div class="score">Score: ${result.score.toFixed(4)} | Method: ${result.metadata?.method || 'unknown'}</div>
                </div>
            `).join('')}
        `;
    } else {
        container.innerHTML = '<p class="placeholder">No results found</p>';
    }
}

// File Upload
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) uploadFile(files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
    const fileType = document.querySelector('input[name="fileType"]:checked').value;
    const status = document.getElementById('uploadStatus');
    status.innerHTML = '<p class="status-loading"><span class="spinner"></span>Uploading...</p>';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', fileType);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            status.innerHTML = `
                <p class="status-success">
                    Upload successful! Indexed ${data.documents_count} document(s).
                    ${data.urls_found ? `Found ${data.urls_found} URLs.` : ''}
                </p>
            `;
        } else {
            status.innerHTML = `<p class="status-error">Upload failed: ${data.message}</p>`;
        }
    } catch (error) {
        status.innerHTML = `<p class="status-error">Error: ${error.message}</p>`;
    }
}

// Crawling
document.getElementById('crawlBtn').addEventListener('click', startCrawl);

async function startCrawl() {
    const urls = document.getElementById('urlList').value
        .split('\n')
        .map(u => u.trim())
        .filter(u => u);

    if (!urls.length) return;

    const useProxy = document.getElementById('useProxy').checked;
    const status = document.getElementById('crawlStatus');
    status.innerHTML = '<p class="status-loading"><span class="spinner"></span>Starting crawl...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/crawl`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ urls, use_proxy: useProxy })
        });

        const data = await response.json();
        status.innerHTML = `
            <p class="status-success">
                Crawl task started. Task ID: ${data.task_id}<br>
                Processing ${data.total_urls} URLs in background.
            </p>
        `;

        // Poll for status
        pollCrawlStatus(data.task_id);
    } catch (error) {
        status.innerHTML = `<p class="status-error">Error: ${error.message}</p>`;
    }
}

async function pollCrawlStatus(taskId) {
    const status = document.getElementById('crawlStatus');

    const poll = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/crawl/${taskId}`);
            const data = await response.json();

            if (data.status === 'completed') {
                const successCount = data.results.filter(r => r.status === 'success').length;
                status.innerHTML = `
                    <p class="status-success">
                        Crawl completed! ${successCount}/${data.results.length} URLs processed successfully.
                    </p>
                `;
            } else {
                setTimeout(poll, 2000);
            }
        } catch (error) {
            status.innerHTML += `<p class="status-error">Error checking status: ${error.message}</p>`;
        }
    };

    setTimeout(poll, 2000);
}

// Documents
document.getElementById('refreshDocs').addEventListener('click', loadDocuments);

async function loadDocuments() {
    const container = document.getElementById('docsList');
    container.innerHTML = '<p class="status-loading"><span class="spinner"></span>Loading...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/documents`);
        const data = await response.json();

        if (data.documents.length === 0) {
            container.innerHTML = '<p class="placeholder">No documents indexed yet</p>';
            return;
        }

        container.innerHTML = data.documents.map(doc => `
            <div class="doc-item">
                <div class="doc-info">
                    <h4>${escapeHtml(doc.title)}</h4>
                    <p>${doc.source_type} | ${doc.chunk_count} chunks | ${new Date(doc.created_at).toLocaleDateString()}</p>
                </div>
                <div class="doc-actions">
                    <button class="btn" onclick="viewDocument('${doc.id}')">View</button>
                    <button class="btn" onclick="deleteDocument('${doc.id}')">Delete</button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `<p class="status-error">Error: ${error.message}</p>`;
    }
}

async function viewDocument(docId) {
    try {
        const response = await fetch(`${API_BASE}/api/documents/${docId}`);
        const doc = await response.json();

        alert(`Title: ${doc.title}\nSource: ${doc.source}\nChunks: ${doc.chunks.length}\n\nContent Preview:\n${doc.content.substring(0, 500)}...`);
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
        const response = await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            loadDocuments();
        } else {
            alert('Failed to delete document');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Settings
document.getElementById('saveProxy').addEventListener('click', saveProxy);
document.getElementById('refreshStats').addEventListener('click', loadStats);

async function saveProxy() {
    const enabled = document.getElementById('proxyEnabled').checked;
    const url = document.getElementById('proxyUrl').value;

    try {
        const response = await fetch(`${API_BASE}/api/config/proxy?enabled=${enabled}&http_proxy=${encodeURIComponent(url)}`, {
            method: 'POST'
        });

        if (response.ok) {
            alert('Proxy settings saved');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function loadStats() {
    const display = document.getElementById('statsDisplay');
    display.innerHTML = '<p class="status-loading"><span class="spinner"></span>Loading...</p>';

    try {
        const [statsRes, configRes] = await Promise.all([
            fetch(`${API_BASE}/api/stats`),
            fetch(`${API_BASE}/api/config`)
        ]);

        const stats = await statsRes.json();
        const config = await configRes.json();

        document.getElementById('proxyEnabled').checked = config.proxy.enabled;
        if (config.proxy.http) {
            document.getElementById('proxyUrl').value = config.proxy.http;
        }

        display.innerHTML = `
            <p><strong>Total Documents:</strong> ${stats.total_documents}</p>
            <p><strong>Total Chunks:</strong> ${stats.total_chunks}</p>
            <p><strong>Vector Index Size:</strong> ${stats.vector_index_size}</p>
            <p><strong>BM25 Index Size:</strong> ${stats.bm25_index_size}</p>
            <p><strong>Embedding Model:</strong> ${config.embedding_model}</p>
            <p><strong>Chunk Sizes:</strong> ${config.chunk_sizes.join(', ')}</p>
        `;
    } catch (error) {
        display.innerHTML = `<p class="status-error">Error: ${error.message}</p>`;
    }
}

// Utility
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initial load
loadStats();
