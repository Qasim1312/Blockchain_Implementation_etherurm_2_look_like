// pkg/network/api.go
package network

import (
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"strconv"
	"sync"
	"time"

	"crypto/sha256"
	"encoding/hex"

	"github.com/gorilla/mux"
	"github.com/qasim/blockchain_assignment_3/pkg/blockchain"
	"github.com/rs/cors"
)

// APIServer represents the HTTP server for blockchain API
type APIServer struct {
	node *Node
	mu   sync.RWMutex
}

// BlockchainInfo is the response format for blockchain data
type BlockchainInfo struct {
	Blocks        []interface{} `json:"blocks"`
	PendingTxs    []interface{} `json:"pendingTransactions"`
	NetworkNodes  []string      `json:"nodes"`
	CurrentHeight uint64        `json:"currentHeight"`
}

// pendingTransactions to store pending transactions
var pendingTransactions []blockchain.Transaction

// NewAPIServer creates a new HTTP API server
func NewAPIServer(node *Node) *APIServer {
	// Initialize pendingTransactions if it's not already initialized
	pendingTransactions = []blockchain.Transaction{}

	return &APIServer{
		node: node,
	}
}

// Start begins the HTTP server
func (api *APIServer) Start(port int) {
	r := mux.NewRouter()

	// CORS handling
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type", "Origin", "Accept"},
		AllowCredentials: true,
	})
	handler := c.Handler(r)

	// Define routes
	r.HandleFunc("/blockchain", api.getBlockchainInfo).Methods("GET")
	r.HandleFunc("/mine", api.mineBlock).Methods("POST")
	r.HandleFunc("/transaction", api.createTransaction).Methods("POST")

	// Start HTTP server
	go func() {
		log.Printf("Starting API server on port %d...", port)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", port), handler); err != nil {
			log.Printf("API server error: %v", err)
		}
	}()
}

// getBlockchainInfo returns all blockchain data
func (api *APIServer) getBlockchainInfo(w http.ResponseWriter, r *http.Request) {
	api.mu.RLock()
	defer api.mu.RUnlock()

	chain := api.node.Chain
	blocks := make([]interface{}, 0)

	for _, block := range chain.Blocks {
		blocks = append(blocks, block)
	}

	pendingTxs := make([]interface{}, 0)
	for _, tx := range pendingTransactions {
		pendingTxs = append(pendingTxs, tx)
	}

	nodes := make([]string, 0)
	for _, peerID := range api.node.Host.Network().Peers() {
		nodes = append(nodes, peerID.String())
	}

	info := BlockchainInfo{
		Blocks:        blocks,
		PendingTxs:    pendingTxs,
		NetworkNodes:  nodes,
		CurrentHeight: chain.GetLatestBlock().Index,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

// mineBlock triggers mining of a new block
func (api *APIServer) mineBlock(w http.ResponseWriter, r *http.Request) {
	api.mu.Lock()
	defer api.mu.Unlock()

	// Trigger block mining in the node
	go api.node.ForceBlockCreation()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "mining_started",
		"message": "Block mining has been triggered",
	})
}

// createTransaction adds a new transaction to the pool
func (api *APIServer) createTransaction(w http.ResponseWriter, r *http.Request) {
	api.mu.Lock()
	defer api.mu.Unlock()

	var tx struct {
		From  string `json:"from"`
		To    string `json:"to"`
		Value string `json:"value"`
		Data  string `json:"data"`
	}

	if err := json.NewDecoder(r.Body).Decode(&tx); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Parse value
	value, err := strconv.ParseUint(tx.Value, 10, 64)
	if err != nil {
		http.Error(w, "Invalid value", http.StatusBadRequest)
		return
	}

	// Create a new transaction
	timestamp := time.Now().Unix()
	txData := []byte(tx.Data)

	// Create transaction ID
	txID := fmt.Sprintf("%s_%s_%d_%d", tx.From, tx.To, value, timestamp)

	// Create the transaction
	newTx := blockchain.Transaction{
		ID:        txID,
		From:      tx.From,
		To:        tx.To,
		Value:     value,
		Data:      txData,
		Timestamp: timestamp,
	}

	// Add transaction to pending transactions
	pendingTransactions = append(pendingTransactions, newTx)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"txID":    newTx.ID,
		"message": "Transaction has been added to the pool",
	})
}

// ForceBlockCreation is a method of Node to trigger block creation
func (n *Node) ForceBlockCreation() {
	log.Println("Manually triggering block creation")

	// Get latest block
	prevBlock := n.Chain.GetLatestBlock()

	// Create a new block with pending transactions
	newBlock := &blockchain.Block{
		Index:        prevBlock.Index + 1,
		Timestamp:    time.Now().Unix(),
		Data:         []byte(fmt.Sprintf("Manually mined block %d", prevBlock.Index+1)),
		PrevHash:     prevBlock.Hash,
		Nonce:        0,
		StateRoot:    n.Chain.CurrentState.GetRootHash(),
		Transactions: pendingTransactions,
	}

	// Mine the block with real proof-of-work
	target := big.NewInt(1)
	target.Lsh(target, 256-24) // Using a fixed difficulty of 24

	for i := uint64(0); i < 1000000; i++ {
		newBlock.Nonce = i
		hash := sha256.Sum256([]byte(fmt.Sprintf("%d%d%s%s%d",
			newBlock.Index, newBlock.Timestamp, newBlock.Data, newBlock.PrevHash, newBlock.Nonce)))

		// Check if hash meets difficulty
		var hashInt big.Int
		hashInt.SetBytes(hash[:])

		// If hash < target, we found a valid block
		if hashInt.Cmp(target) == -1 {
			newBlock.Hash = hex.EncodeToString(hash[:])
			log.Printf("[MINING] Valid nonce found: %d", newBlock.Nonce)
			break
		}
	}

	// Add to chain
	n.Chain.Blocks = append(n.Chain.Blocks, newBlock)

	// Update state with new transactions
	for _, tx := range newBlock.Transactions {
		n.Chain.CurrentState.ApplyTransaction(&tx)
	}

	// Reset pending transactions
	pendingTransactions = []blockchain.Transaction{}

	// Broadcast the block
	blockData, _ := json.Marshal(newBlock)
	n.PubSub.Publish(BlocksTopic, blockData)

	log.Printf("Manually mined block: height=%d hash=%s nonce=%d",
		newBlock.Index, newBlock.Hash, newBlock.Nonce)
}
