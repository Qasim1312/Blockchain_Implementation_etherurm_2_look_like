// app.js
// Configuration
const NODE_ENDPOINTS = [
    { name: "Node 1 (Port 4001)", url: "http://localhost:8080" },
    { name: "Node 2 (Port 4002)", url: "http://localhost:8081" },
    { name: "Node 3 (Port 4003)", url: "http://localhost:8082" }
  ];
  let currentNodeIndex = 0;
  const REFRESH_INTERVAL = 5000; // 5 seconds
  
  // Application state
  let connected = false;
  let blockchain = { blocks: [], nodes: [], currentHeight: 0 };
  
  // DOM elements
  const statusIndicator    = document.getElementById('status-indicator');
  const connectionText     = document.getElementById('connection-text');
  const chainHeightEl      = document.getElementById('chain-height');
  const nodeCountEl        = document.getElementById('node-count');
  const latestNonceEl      = document.getElementById('latest-nonce');
  const blockchainDisplay  = document.getElementById('blockchain-display');
  const nodesDisplay       = document.getElementById('nodes-display');
  const refreshBtn         = document.getElementById('refresh-btn');
  const nodeSelector       = document.getElementById('node-selector');
  const currentNodeEl      = document.getElementById('current-node');
  
  // Initialize app
  function init() {
    setupEventListeners();
    setupNodeSelector();
    refreshBlockchainData();
    setInterval(refreshBlockchainData, REFRESH_INTERVAL);
  }
  
  // Node selector
  function setupNodeSelector() {
    nodeSelector.innerHTML = '';
    NODE_ENDPOINTS.forEach((node, idx) => {
      const opt = document.createElement('option');
      opt.value = idx;
      opt.textContent = node.name;
      nodeSelector.appendChild(opt);
    });
    updateCurrentNodeDisplay();
  }
  
  function switchNode(index) {
    currentNodeIndex = index;
    updateCurrentNodeDisplay();
    refreshBlockchainData();
  }
  
  function updateCurrentNodeDisplay() {
    currentNodeEl.textContent = NODE_ENDPOINTS[currentNodeIndex].name;
  }
  
  // Event listeners
  function setupEventListeners() {
    refreshBtn.addEventListener('click', refreshBlockchainData);
    nodeSelector.addEventListener('change', e => switchNode(+e.target.value));
  }
  
  // Fetching data
  async function fetchBlockchainData() {
    try {
      const nodeConfig = NODE_ENDPOINTS[currentNodeIndex];
      const resp = await fetch(`${nodeConfig.url}/blockchain`);
      if (!resp.ok) throw new Error('Network response was not ok');
      return await resp.json();
    } catch (err) {
      console.error('Error fetching blockchain data:', err);
      return null;
    }
  }
  
  // Main refresh
  async function refreshBlockchainData() {
    const data = await fetchBlockchainData();
    if (data) {
      connected = true;
      blockchain = data;
      updateConnectionStatus(true);
      updateStats();
      renderBlockchain();
      renderNetworkNodes();
    } else {
      connected = false;
      updateConnectionStatus(false);
    }
  }
  
  function updateConnectionStatus(isConnected) {
    if (isConnected) {
      statusIndicator.classList.add('connected');
      connectionText.textContent = `Connected to ${NODE_ENDPOINTS[currentNodeIndex].name}`;
    } else {
      statusIndicator.classList.remove('connected');
      connectionText.textContent = `Disconnected from ${NODE_ENDPOINTS[currentNodeIndex].name}`;
    }
  }
  
  function updateStats() {
    chainHeightEl.textContent = blockchain.currentHeight;
    nodeCountEl.textContent   = blockchain.nodes.length;
    
    // Display latest block nonce if blocks exist
    if (blockchain.blocks && blockchain.blocks.length > 0) {
      const latestBlock = blockchain.blocks[blockchain.blocks.length - 1];
      latestNonceEl.textContent = latestBlock.Nonce || '-';
    } else {
      latestNonceEl.textContent = '-';
    }
  }
  
  // Rendering
  function renderBlockchain() {
    if (!blockchain.blocks.length) {
      blockchainDisplay.innerHTML = '<div class="loading">No blocks in the blockchain</div>';
      return;
    }
    blockchainDisplay.innerHTML = '';
    blockchain.blocks.forEach(b => {
      const el = document.createElement('div');
      el.className = 'block';
      el.innerHTML = `
        <div class="block-header">Block #${b.Index}</div>
        <div class="block-data">Hash: ${truncateString(b.Hash, 8)}</div>
        <div class="block-data">Prev: ${truncateString(b.PrevHash, 8)}</div>
        <div class="block-data">Time: ${formatTimestamp(b.Timestamp)}</div>
        <div class="block-data nonce-value">Nonce: <span class="highlight-nonce">${b.Nonce || 'N/A'}</span></div>
        <div class="block-data">Txs: ${b.Transactions?.length || 0}</div>
      `;
      blockchainDisplay.appendChild(el);
    });
  }
  
  function renderNetworkNodes() {
    if (!blockchain.nodes.length) {
      nodesDisplay.innerHTML = '<div class="loading">No nodes connected</div>';
      return;
    }
    nodesDisplay.innerHTML = '';
    blockchain.nodes.forEach(id => {
      const el = document.createElement('div');
      el.className = 'node';
      el.innerHTML = `
        <div class="node-id">Node ID: ${truncateString(id, 12)}</div>
        <div class="node-details">Status: Active</div>
      `;
      nodesDisplay.appendChild(el);
    });
  }
  
  function truncateString(str, len) {
    return str?.length > len ? str.slice(0, len) + '...' : (str || '');
  }
  
  function formatTimestamp(ts) {
    return new Date(ts * 1000).toLocaleTimeString();
  }
  
  window.addEventListener('DOMContentLoaded', init);
  