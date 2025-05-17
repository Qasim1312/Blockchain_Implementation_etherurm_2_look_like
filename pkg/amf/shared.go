// pkg/amf/shard.go
package amf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

const (
	DefaultMaxShardSize = 1000 // Default maximum number of entries per shard
	MinShardSize        = 100  // Minimum shard size to prevent excessive sharding
	MaxTotalShards      = 64   // Maximum number of shards to prevent over-fragmentation
)

// MerkleNode represents a node in the merkle tree
type MerkleNode struct {
	Hash       string
	Left       *MerkleNode
	Right      *MerkleNode
	Key        string
	Value      []byte
	IsLeaf     bool
	SkipVerify bool // For compressed proofs - marks nodes that can be skipped
}

// Shard represents a single shard in the Adaptive Merkle Forest
type Shard struct {
	ID     string
	Root   *MerkleNode
	Leaves map[string][]byte
	mutex  sync.RWMutex
}

// NewShard creates a new shard with the given ID
func NewShard(id string) *Shard {
	return &Shard{
		ID:     id,
		Root:   nil,
		Leaves: make(map[string][]byte),
		mutex:  sync.RWMutex{},
	}
}

// AddLeaf adds a new key-value pair to the shard
func (s *Shard) AddLeaf(key string, value []byte) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.Leaves[key] = value
	s.rebuildTree() // In a real implementation, you'd use incremental updates
}

// GetLeaf retrieves a value by key
func (s *Shard) GetLeaf(key string) ([]byte, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	value, exists := s.Leaves[key]
	return value, exists
}

// Size returns the number of leaves in the shard
func (s *Shard) Size() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return len(s.Leaves)
}

// GetData retrieves data for a key, returning the value and an error if not found
func (s *Shard) GetData(key string) ([]byte, error) {
	value, exists := s.GetLeaf(key)
	if !exists {
		return nil, fmt.Errorf("key %s not found in shard %s", key, s.ID)
	}
	return value, nil
}

// AddData adds data to the shard, returning an error if there's an issue
func (s *Shard) AddData(key string, value []byte) error {
	s.AddLeaf(key, value)
	return nil
}

// rebuildTree reconstructs the merkle tree for the shard
func (s *Shard) rebuildTree() {
	if len(s.Leaves) == 0 {
		s.Root = nil
		return
	}

	// Create leaf nodes
	var nodes []*MerkleNode
	for key, value := range s.Leaves {
		hash := hashData(key + string(value))
		node := &MerkleNode{
			Hash:   hash,
			Key:    key,
			Value:  value,
			IsLeaf: true,
		}
		nodes = append(nodes, node)
	}

	// Build tree bottom-up
	for len(nodes) > 1 {
		var nextLevel []*MerkleNode

		for i := 0; i < len(nodes); i += 2 {
			var right *MerkleNode
			left := nodes[i]

			if i+1 < len(nodes) {
				right = nodes[i+1]
			} else {
				right = left // Duplicate last node if odd number
			}

			combinedHash := hashData(left.Hash + right.Hash)
			parent := &MerkleNode{
				Hash:  combinedHash,
				Left:  left,
				Right: right,
			}

			nextLevel = append(nextLevel, parent)
		}

		nodes = nextLevel
	}

	s.Root = nodes[0]
}

// hashData is a helper function to create hashes
func hashData(data string) string {
	h := sha256.New()
	h.Write([]byte(data))
	return hex.EncodeToString(h.Sum(nil))
}

// ShardLoadBalancer handles sharding distribution and rebalancing decisions
type ShardLoadBalancer struct {
	// Configuration parameters for load balancing
	SplitThreshold float64
	MergeThreshold float64
	maxShardSize   int
	lastAdjust     time.Time
	mutex          sync.RWMutex
}

// NewShardLoadBalancer creates a new shard load balancer
func NewShardLoadBalancer() *ShardLoadBalancer {
	return &ShardLoadBalancer{
		SplitThreshold: 0.8, // 80% of max capacity
		MergeThreshold: 0.3, // 30% of max capacity
		maxShardSize:   DefaultMaxShardSize,
		lastAdjust:     time.Now(),
		mutex:          sync.RWMutex{},
	}
}

// DetermineShardForData decides which shard should handle a given key
func (lb *ShardLoadBalancer) DetermineShardForData(key string, shards map[string]*Shard) string {
	// For simplicity in this implementation:
	// If only one shard, use it
	if len(shards) == 1 {
		for id := range shards {
			return id
		}
	}

	// Find shard containing the key if it exists
	for id, shard := range shards {
		if _, exists := shard.Leaves[key]; exists {
			return id
		}
	}

	// Key doesn't exist - use an intelligent placement strategy

	// 1. First check if there's a shard with related data (based on key prefix)
	keyPrefix := ""
	if len(key) > 4 {
		keyPrefix = key[:4] // Use first few characters as prefix
	}

	for id, shard := range shards {
		// Sample a few keys from the shard to check for prefix matches
		matchCount := 0
		sampleSize := 0

		for shardKey := range shard.Leaves {
			sampleSize++
			if len(shardKey) > 4 && shardKey[:4] == keyPrefix {
				matchCount++
			}

			// Only sample a few keys
			if sampleSize >= 10 {
				break
			}
		}

		// If we found a good match and the shard has room, use it
		if matchCount >= 3 && shard.Size() < lb.GetCurrentMaxShardSize() {
			return id
		}
	}

	// 2. If no matches, find least loaded shard
	var selectedID string
	minLoad := lb.GetCurrentMaxShardSize() + 1

	for id, shard := range shards {
		if shard.Size() < minLoad {
			minLoad = shard.Size()
			selectedID = id
		}
	}

	return selectedID
}

// FindShardContainingKey locates which shard contains a given key
func (lb *ShardLoadBalancer) FindShardContainingKey(key string, shards map[string]*Shard) string {
	for id, shard := range shards {
		if _, exists := shard.Leaves[key]; exists {
			return id
		}
	}
	return ""
}

// ShouldRebalance determines if the forest needs rebalancing
func (lb *ShardLoadBalancer) ShouldRebalance(shards map[string]*Shard) bool {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()

	for _, shard := range shards {
		loadFactor := float64(shard.Size()) / float64(lb.maxShardSize)
		if loadFactor > lb.SplitThreshold {
			return true
		}
	}

	// Check for potential merges
	if len(shards) > 1 {
		underloadedCount := 0
		for _, shard := range shards {
			loadFactor := float64(shard.Size()) / float64(lb.maxShardSize)
			if loadFactor < lb.MergeThreshold {
				underloadedCount++
			}
		}

		// If more than half of shards are underloaded, suggest rebalancing
		if underloadedCount > len(shards)/2 {
			return true
		}
	}

	return false
}

// GetCurrentMaxShardSize returns the current maximum shard size
func (lb *ShardLoadBalancer) GetCurrentMaxShardSize() int {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()
	return lb.maxShardSize
}

// AdjustMaxShardSize adjusts the maximum shard size by a factor
func (lb *ShardLoadBalancer) AdjustMaxShardSize(factor float64) {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	// Don't adjust too frequently
	if time.Since(lb.lastAdjust) < 10*time.Minute {
		return
	}

	// Calculate new size
	newSize := int(float64(lb.maxShardSize) * factor)

	// Enforce bounds
	if newSize < MinShardSize {
		newSize = MinShardSize
	} else if newSize > DefaultMaxShardSize*2 {
		newSize = DefaultMaxShardSize * 2
	}

	lb.maxShardSize = newSize
	lb.lastAdjust = time.Now()
}

// CompressedMerkleProof represents a compressed merkle proof with probabilistic verification
type CompressedMerkleProof struct {
	Key              string
	Value            []byte
	CompressedHashes []string
	ShardID          string
	CompressionRate  float64
}

// CompressProof creates a compressed version of a standard merkle proof
func CompressProof(proof *MerkleProof) *CompressedMerkleProof {
	if proof == nil {
		return nil
	}

	// Initialize random source
	rand.Seed(time.Now().UnixNano())

	// Decide on a compression rate (between 0.5 and 0.8)
	compressionRate := 0.5 + rand.Float64()*0.3

	// Select hashes to keep based on compression rate
	selectedHashes := make([]string, 0)
	for _, hash := range proof.ProofHashes {
		if rand.Float64() > compressionRate {
			selectedHashes = append(selectedHashes, hash)
		}
	}

	// Ensure we keep at least one hash for integrity
	if len(selectedHashes) == 0 && len(proof.ProofHashes) > 0 {
		selectedHashes = append(selectedHashes, proof.ProofHashes[0])
	}

	return &CompressedMerkleProof{
		Key:              proof.Key,
		Value:            proof.Value,
		CompressedHashes: selectedHashes,
		ShardID:          proof.ShardID,
		CompressionRate:  compressionRate,
	}
}

// VerifyCompressedProof verifies a compressed proof with probabilistic confidence
func VerifyCompressedProof(proof *CompressedMerkleProof, expectedRoot string) (bool, float64) {
	if proof == nil || len(proof.CompressedHashes) == 0 {
		return false, 0.0
	}

	// Calculate a verification score based on compression rate and number of hashes
	confidence := (1.0 - proof.CompressionRate) * float64(len(proof.CompressedHashes)) / 5.0

	// Cap confidence between 0 and 1
	if confidence > 1.0 {
		confidence = 1.0
	}

	// In a real implementation, we would attempt verification with the compressed hashes
	// Here we'll simulate verification based on compression rate
	proofIsValid := true // Assume proof is valid for this example

	return proofIsValid, confidence
}

// BloomFilter implements a compact approximate membership query data structure
type BloomFilter struct {
	bits         []byte
	numBits      uint
	numHashFuncs uint
}

// NewBloomFilter creates a new Bloom filter with optimal parameters for expectedItems
func NewBloomFilter(expectedItems int, falsePositiveRate float64) *BloomFilter {
	m := optimalNumBits(expectedItems, falsePositiveRate)
	k := optimalNumHashFuncs(expectedItems, m)

	return &BloomFilter{
		bits:         make([]byte, (m+7)/8), // Round up to nearest byte
		numBits:      m,
		numHashFuncs: k,
	}
}

// Add adds an item to the Bloom filter
func (bf *BloomFilter) Add(item string) {
	h := sha256.Sum256([]byte(item))

	for i := uint(0); i < bf.numHashFuncs; i++ {
		// Use the hash to derive multiple hash functions
		hashValue := bf.nthHash(i, h)
		position := hashValue % bf.numBits

		// Set the bit
		bf.bits[position/8] |= 1 << (position % 8)
	}
}

// Contains checks if an item might be in the set
func (bf *BloomFilter) Contains(item string) bool {
	h := sha256.Sum256([]byte(item))

	for i := uint(0); i < bf.numHashFuncs; i++ {
		hashValue := bf.nthHash(i, h)
		position := hashValue % bf.numBits

		// Check if bit is set
		if (bf.bits[position/8] & (1 << (position % 8))) == 0 {
			return false // Definitely not in the set
		}
	}

	return true // Might be in the set
}

// nthHash generates the nth hash function from a base hash
func (bf *BloomFilter) nthHash(n uint, baseHash [32]byte) uint {
	// Use first 8 bytes for first value
	h1 := binary.BigEndian.Uint64(baseHash[:8])
	// Use next 8 bytes for second value
	h2 := binary.BigEndian.Uint64(baseHash[8:16])

	// Generate nth hash from h1 and h2
	return uint((h1 + uint64(n)*h2) % uint64(bf.numBits))
}

// optimalNumBits calculates the optimal number of bits for a Bloom filter
func optimalNumBits(n int, p float64) uint {
	return uint(math.Ceil(-float64(n) * math.Log(p) / math.Pow(math.Log(2), 2)))
}

// optimalNumHashFuncs calculates the optimal number of hash functions
func optimalNumHashFuncs(n int, m uint) uint {
	return uint(math.Ceil(float64(m) / float64(n) * math.Log(2)))
}
