// pkg/blockchain/block.go
package blockchain

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
)

// Block represents a block in the blockchain
type Block struct {
	Index        uint64
	Timestamp    int64
	Data         []byte
	PrevHash     string
	Hash         string
	Nonce        uint64
	StateRoot    string
	Transactions []Transaction
	Difficulty   uint64
	Validator    string
	Signature    []byte
	// Advanced block structure enhancements
	AccumulatorValue   []byte   // Cryptographic accumulator for compact state representation
	MultiLevelMerkle   *MLMTree // Multi-level Merkle tree structure
	EntropyValidation  float64  // Entropy-based validation metric
	CompressedStateRef string   // Reference to compressed state archive
}

// Transaction represents a blockchain transaction
type Transaction struct {
	ID        string
	From      string
	To        string
	Value     uint64
	Data      []byte
	Signature []byte
	Timestamp int64
}

// MLMTree represents a Multi-Level Merkle Tree for advanced state representation
type MLMTree struct {
	Levels        int               // Number of levels in the tree
	RootHash      string            // Root hash of the tree
	LevelHashes   map[int][]string  // Hashes at each level
	OptimalProofs map[string][]byte // Optimized proofs for key paths
	Metadata      map[string]string // Additional metadata for tree navigation
}

// NewMultiLevelMerkleTree creates a new multi-level Merkle tree
func NewMultiLevelMerkleTree(height int) *MLMTree {
	return &MLMTree{
		Levels:        height,
		LevelHashes:   make(map[int][]string),
		OptimalProofs: make(map[string][]byte),
		Metadata:      make(map[string]string),
	}
}

// InsertDataIntoMLM inserts data into the multi-level Merkle tree
func (m *MLMTree) InsertData(key string, value []byte) {
	// Hash the key-value pair
	h := sha256.New()
	h.Write([]byte(key))
	h.Write(value)
	leafHash := hex.EncodeToString(h.Sum(nil))

	// Insert at bottom level
	bottomLevel := m.Levels - 1
	if _, exists := m.LevelHashes[bottomLevel]; !exists {
		m.LevelHashes[bottomLevel] = make([]string, 0)
	}

	m.LevelHashes[bottomLevel] = append(m.LevelHashes[bottomLevel], leafHash)

	// Regenerate tree structure
	m.rebuildTree()

	// Store metadata for quick lookups
	m.Metadata[key] = leafHash
}

// rebuildTree reconstructs the multi-level Merkle tree from bottom up
func (m *MLMTree) rebuildTree() {
	// Start from the bottom level and work up
	for level := m.Levels - 1; level > 0; level-- {
		currentLevelHashes := m.LevelHashes[level]
		upperLevel := level - 1

		// Create upper level if it doesn't exist
		if _, exists := m.LevelHashes[upperLevel]; !exists {
			m.LevelHashes[upperLevel] = make([]string, 0)
		} else {
			// Clear existing hashes at this level since we're rebuilding
			m.LevelHashes[upperLevel] = make([]string, 0)
		}

		// Combine pairs of hashes to form the upper level
		for i := 0; i < len(currentLevelHashes); i += 2 {
			var combinedHash string

			if i+1 < len(currentLevelHashes) {
				// Hash two children together
				h := sha256.New()
				h.Write([]byte(currentLevelHashes[i] + currentLevelHashes[i+1]))
				combinedHash = hex.EncodeToString(h.Sum(nil))
			} else {
				// Odd node out, promote it up
				combinedHash = currentLevelHashes[i]
			}

			m.LevelHashes[upperLevel] = append(m.LevelHashes[upperLevel], combinedHash)
		}
	}

	// Set the root hash
	if len(m.LevelHashes[0]) > 0 {
		m.RootHash = m.LevelHashes[0][0]
	}
}

// GenerateOptimalProof creates an optimized proof for a key
func (m *MLMTree) GenerateOptimalProof(key string) []byte {
	// Create proof by gathering sibling hashes along path
	// This is a simplified implementation
	if leafHash, exists := m.Metadata[key]; exists {
		h := sha256.New()
		h.Write([]byte(key))
		h.Write([]byte(leafHash))
		m.OptimalProofs[key] = h.Sum(nil)
		return m.OptimalProofs[key]
	}

	return nil
}

// CryptoAccumulator represents a cryptographic accumulator for efficient state verification
type CryptoAccumulator struct {
	Value      []byte
	Modulus    []byte
	Witnesses  map[string][]byte
	Generator  []byte
	Operations []string
}

// NewCryptoAccumulator creates a new cryptographic accumulator
func NewCryptoAccumulator() *CryptoAccumulator {
	// Simplified implementation of a cryptographic accumulator
	return &CryptoAccumulator{
		Value:      make([]byte, 32), // 256-bit accumulator value
		Witnesses:  make(map[string][]byte),
		Operations: make([]string, 0),
	}
}

// AddElement adds an element to the accumulator
func (ca *CryptoAccumulator) AddElement(data []byte) {
	// In a real implementation, this would use RSA accumulation or similar
	// Here we use a simplified approach for demonstration
	h := sha256.New()
	h.Write(ca.Value)
	h.Write(data)
	ca.Value = h.Sum(nil)

	// Record operation
	ca.Operations = append(ca.Operations, "add:"+hex.EncodeToString(data))

	// Update witnesses (simplified)
	for key := range ca.Witnesses {
		h = sha256.New()
		h.Write(ca.Witnesses[key])
		h.Write(data)
		ca.Witnesses[key] = h.Sum(nil)
	}
}

// GenerateWitness creates a witness for proving membership
func (ca *CryptoAccumulator) GenerateWitness(data []byte) []byte {
	// Generate a witness for the element (simplified implementation)
	dataStr := hex.EncodeToString(data)

	if _, exists := ca.Witnesses[dataStr]; !exists {
		// Create a new witness
		h := sha256.New()
		h.Write(ca.Value)
		h.Write(data)
		witness := h.Sum(nil)
		ca.Witnesses[dataStr] = witness
	}

	return ca.Witnesses[dataStr]
}

// VerifyMembership verifies that data is a member of the accumulator
func (ca *CryptoAccumulator) VerifyMembership(data []byte, witness []byte) bool {
	// Verify membership using the witness (simplified implementation)
	h := sha256.New()
	h.Write(witness)
	h.Write(data)
	verification := h.Sum(nil)

	// Compare with accumulator value
	for i := 0; i < len(verification); i++ {
		if verification[i] != ca.Value[i] {
			return false
		}
	}

	return true
}

// CalculateBlockEntropy calculates the entropy for block validation
func CalculateBlockEntropy(block *Block) float64 {
	// A simplified entropy calculation for blocks
	// In a real implementation, this would use Shannon entropy
	// across the distribution of transaction values, nonce bits, etc.

	// Count unique transaction senders and receivers
	uniqueAddresses := make(map[string]bool)
	for _, tx := range block.Transactions {
		uniqueAddresses[tx.From] = true
		uniqueAddresses[tx.To] = true
	}

	// Basic entropy metric
	return float64(len(uniqueAddresses)) / float64(2*len(block.Transactions)+1)
}

// String converts a block to a string for hashing
func (b *Block) String() string {
	return fmt.Sprintf("%d%d%s%s%d", b.Index, b.Timestamp, string(b.Data), b.PrevHash, b.Nonce)
}

// CalculateHash calculates the hash of the block
func (b *Block) CalculateHash() string {
	// Use the String() method to get a consistent string representation
	record := b.String()
	h := sha256.New()
	h.Write([]byte(record))
	hashed := h.Sum(nil)
	return hex.EncodeToString(hashed)
}

// NewBlock creates and returns a new block
func NewBlock(index uint64, data []byte, prevHash string, transactions []Transaction) *Block {
	block := &Block{
		Index:        index,
		Timestamp:    time.Now().Unix(),
		Data:         data,
		PrevHash:     prevHash,
		Nonce:        0,
		Transactions: transactions,
		Difficulty:   0,
		// Initialize advanced block components
		MultiLevelMerkle:  NewMultiLevelMerkleTree(4), // 4 levels by default
		AccumulatorValue:  make([]byte, 32),
		EntropyValidation: 0,
	}

	block.Hash = block.CalculateHash()
	return block
}
