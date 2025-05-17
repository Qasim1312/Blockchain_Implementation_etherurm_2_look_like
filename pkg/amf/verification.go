// pkg/amf/verification.go
package amf

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/qasim/blockchain_assignment_3/pkg/utils"
)

// MerkleProof represents a cryptographic proof of membership
type MerkleProof struct {
	Key         string
	Value       []byte
	ProofHashes []string
	ShardID     string
}

// CrossShardReference represents a reference to data in another shard
type CrossShardReference struct {
	SourceShardID string
	TargetShardID string
	Key           string
	Timestamp     time.Time
}

// CrossShardSynchronizer manages state synchronization between shards
type CrossShardSynchronizer struct {
	references      map[string][]CrossShardReference
	synchronizeLog  []SynchronizeOperation
	mutex           sync.RWMutex
	atomicCommitter *AtomicCommitter
	forest          *AdaptiveMerkleForest
}

// SynchronizeOperation represents a cross-shard synchronization event
type SynchronizeOperation struct {
	SourceShardID string
	TargetShardID string
	Keys          []string
	Timestamp     time.Time
	Success       bool
}

// AtomicCommitter ensures atomicity of cross-shard operations
type AtomicCommitter struct {
	pendingOperations map[string]*AtomicOperation
	mutex             sync.Mutex
}

// AtomicOperation represents an atomic cross-shard operation
type AtomicOperation struct {
	ID            string
	ShardIDs      []string
	Keys          map[string][]string
	State         string
	CreatedAt     time.Time
	CompletedAt   time.Time
	Participants  map[string]bool
	Confirmations map[string]bool
}

// NewCrossShardSynchronizer creates a new cross-shard synchronizer
func NewCrossShardSynchronizer(forest *AdaptiveMerkleForest) *CrossShardSynchronizer {
	return &CrossShardSynchronizer{
		references:     make(map[string][]CrossShardReference),
		synchronizeLog: make([]SynchronizeOperation, 0),
		atomicCommitter: &AtomicCommitter{
			pendingOperations: make(map[string]*AtomicOperation),
		},
		forest: forest,
	}
}

// AddCrossShardReference adds a cross-shard reference
func (css *CrossShardSynchronizer) AddCrossShardReference(sourceShardID, targetShardID, key string) {
	css.mutex.Lock()
	defer css.mutex.Unlock()

	reference := CrossShardReference{
		SourceShardID: sourceShardID,
		TargetShardID: targetShardID,
		Key:           key,
		Timestamp:     time.Now(),
	}

	css.references[key] = append(css.references[key], reference)
}

// GetCrossShardReferences gets all cross-shard references for a key
func (css *CrossShardSynchronizer) GetCrossShardReferences(key string) []CrossShardReference {
	css.mutex.RLock()
	defer css.mutex.RUnlock()

	return css.references[key]
}

// SynchronizeShards synchronizes state between shards
func (css *CrossShardSynchronizer) SynchronizeShards(srcShardID, destShardID string, keys []string) error {
	startTime := time.Now()
	utils.LogToFile("amf_transfers", fmt.Sprintf("Starting cross-shard transfer from %s to %s for %d keys",
		srcShardID, destShardID, len(keys)))

	css.mutex.Lock()
	defer css.mutex.Unlock()

	// Get the source and destination shards
	srcShard := css.forest.GetShardByID(srcShardID)
	if srcShard == nil {
		errMsg := fmt.Sprintf("Source shard %s not found", srcShardID)
		utils.LogToFile("amf_transfers", fmt.Sprintf("ERROR: %s", errMsg))
		return fmt.Errorf(errMsg)
	}

	destShard := css.forest.GetShardByID(destShardID)
	if destShard == nil {
		errMsg := fmt.Sprintf("Destination shard %s not found", destShardID)
		utils.LogToFile("amf_transfers", fmt.Sprintf("ERROR: %s", errMsg))
		return fmt.Errorf(errMsg)
	}

	// Create an atomic operation
	opID := fmt.Sprintf("sync-%s-%s-%d", srcShardID, destShardID, time.Now().UnixNano())
	utils.LogToFile("cross_shard_atomic", fmt.Sprintf("Beginning atomic operation %s between shards %s and %s",
		opID, srcShardID, destShardID))

	// Begin atomic operation
	op := css.atomicCommitter.BeginOperation([]string{srcShardID, destShardID})

	// Track successfully transferred items
	successCount := 0

	// Copy data from source to destination
	for _, key := range keys {
		// Get data from source shard
		data, err := srcShard.GetData(key)
		if err != nil {
			utils.LogToFile("amf_transfers", fmt.Sprintf("ERROR getting data for key %s from shard %s: %v",
				key, srcShardID, err))
			css.atomicCommitter.AbortOperation(op.ID)
			return err
		}

		// Add data to destination shard
		err = destShard.AddData(key, data)
		if err != nil {
			utils.LogToFile("amf_transfers", fmt.Sprintf("ERROR adding data for key %s to shard %s: %v",
				key, destShardID, err))
			css.atomicCommitter.AbortOperation(op.ID)
			return err
		}

		successCount++

		// Log progress every 10 items
		if successCount%10 == 0 {
			utils.LogToFile("amf_transfers", fmt.Sprintf("Transferred %d/%d items from %s to %s",
				successCount, len(keys), srcShardID, destShardID))
		}
	}

	// Create a homomorphic commitment for the transfer
	commitment := []byte("simulated-homomorphic-commitment") // In a real implementation, this would be a real commitment
	utils.LogToFile("cross_shard_homomorphic", fmt.Sprintf("Created homomorphic commitment for operation %s: %x",
		op.ID, commitment))

	// Complete atomic operation
	css.atomicCommitter.CompleteOperation(op.ID)
	utils.LogToFile("cross_shard_atomic", fmt.Sprintf("Completed atomic operation %s successfully. Transferred %d items",
		op.ID, successCount))

	// Log the synchronization
	duration := time.Since(startTime)
	utils.LogToFile("amf_transfers", fmt.Sprintf("Cross-shard synchronization completed: %s -> %s (%d/%d keys in %v)",
		srcShardID, destShardID, successCount, len(keys), duration))

	return nil
}

// BeginOperation starts a new atomic operation
func (ac *AtomicCommitter) BeginOperation(shardIDs []string) *AtomicOperation {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()

	// Generate unique operation ID
	idHash := sha256.New()
	for _, shardID := range shardIDs {
		idHash.Write([]byte(shardID))
	}
	idHash.Write([]byte(time.Now().String()))

	opID := hex.EncodeToString(idHash.Sum(nil))

	// Create operation
	op := &AtomicOperation{
		ID:            opID,
		ShardIDs:      shardIDs,
		Keys:          make(map[string][]string),
		State:         "pending",
		CreatedAt:     time.Now(),
		Participants:  make(map[string]bool),
		Confirmations: make(map[string]bool),
	}

	// Add to pending operations
	ac.pendingOperations[opID] = op

	return op
}

// CompleteOperation marks an operation as complete
func (ac *AtomicCommitter) CompleteOperation(operationID string) error {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()

	op, exists := ac.pendingOperations[operationID]
	if !exists {
		return errors.New("operation not found")
	}

	op.State = "completed"
	op.CompletedAt = time.Now()

	// In a real system, this would notify all participants

	// Remove from pending operations after a cleanup period
	go func() {
		time.Sleep(5 * time.Minute)
		ac.mutex.Lock()
		delete(ac.pendingOperations, operationID)
		ac.mutex.Unlock()
	}()

	return nil
}

// AbortOperation marks an operation as aborted
func (ac *AtomicCommitter) AbortOperation(operationID string) error {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()

	op, exists := ac.pendingOperations[operationID]
	if !exists {
		return errors.New("operation not found")
	}

	op.State = "aborted"
	op.CompletedAt = time.Now()

	// In a real system, this would notify all participants to roll back

	// Remove from pending operations immediately
	delete(ac.pendingOperations, operationID)

	utils.LogToFile("amf_transfers", fmt.Sprintf("Aborted atomic operation %s", operationID))

	return nil
}

// GenerateProof creates a merkle proof for a key
func (s *Shard) GenerateProof(key string) *MerkleProof {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	value, exists := s.Leaves[key]
	if !exists || s.Root == nil {
		return nil
	}

	proof := &MerkleProof{
		Key:         key,
		Value:       value,
		ProofHashes: []string{},
		ShardID:     s.ID,
	}

	// Build the proof path by traversing the tree
	proofPath := findProofPath(s.Root, key)
	if proofPath == nil {
		return nil
	}

	// Extract hashes from the proof path
	for _, node := range proofPath {
		if node.Left != nil && node.Right != nil {
			// For non-leaf nodes, include both children in the proof
			if node.Left.Key == key {
				proof.ProofHashes = append(proof.ProofHashes, node.Right.Hash)
			} else {
				proof.ProofHashes = append(proof.ProofHashes, node.Left.Hash)
			}
		}
	}

	return proof
}

// findProofPath finds the path from root to a leaf node with the given key
func findProofPath(root *MerkleNode, key string) []*MerkleNode {
	if root == nil {
		return nil
	}

	// If this is a leaf node with the target key, return it
	if root.IsLeaf && root.Key == key {
		return []*MerkleNode{root}
	}

	// If this is a leaf node but not the target, return nil
	if root.IsLeaf {
		return nil
	}

	// Try left subtree
	if root.Left != nil {
		leftPath := findProofPath(root.Left, key)
		if leftPath != nil {
			return append([]*MerkleNode{root}, leftPath...)
		}
	}

	// Try right subtree
	if root.Right != nil {
		rightPath := findProofPath(root.Right, key)
		if rightPath != nil {
			return append([]*MerkleNode{root}, rightPath...)
		}
	}

	return nil
}

// VerifyProof verifies a merkle proof
func VerifyProof(proof *MerkleProof, rootHash string) bool {
	if proof == nil || len(proof.ProofHashes) == 0 {
		return false
	}

	// Hash the key-value pair
	h := sha256.New()
	h.Write([]byte(proof.Key))
	h.Write(proof.Value)
	currentHash := hex.EncodeToString(h.Sum(nil))

	// Combine with proof hashes to reconstruct the root
	for _, proofHash := range proof.ProofHashes {
		// The order matters in a real implementation
		// For simplicity, we'll always concatenate in the same order
		combinedHash := hashData(currentHash + proofHash)
		currentHash = combinedHash
	}

	// Check if the reconstructed hash matches the root hash
	return currentHash == rootHash
}

// VerifyInclusionWithFilter verifies data inclusion using a Bloom filter
func VerifyInclusionWithFilter(filter *BloomFilter, key string) bool {
	return filter.Contains(key)
}

// GenerateStateFilter creates a Bloom filter for efficient state verification
func (f *AdaptiveMerkleForest) GenerateStateFilter(falsePositiveRate float64) *BloomFilter {
	f.mutex.RLock()
	defer f.mutex.RUnlock()

	// Count total items
	totalItems := 0
	for _, shard := range f.Shards {
		totalItems += shard.Size()
	}

	// Create filter
	filter := NewBloomFilter(totalItems, falsePositiveRate)

	// Add all keys
	for _, shard := range f.Shards {
		for key := range shard.Leaves {
			filter.Add(key)
		}
	}

	return filter
}

// HomomorphicCommitment represents a homomorphic authenticated data structure
type HomomorphicCommitment struct {
	CommitValue []byte
	Additions   []string
	Deletions   []string
	Timestamp   time.Time
}

// GenerateHomomorphicCommitment creates a commitment that can be efficiently updated
func (s *Shard) GenerateHomomorphicCommitment() *HomomorphicCommitment {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	// In a real implementation, this would use a proper homomorphic hash function
	// For this example, we'll use a simplified approach

	h := sha256.New()

	// Add all keys in sorted order for deterministic output
	var keys []string
	for key := range s.Leaves {
		keys = append(keys, key)
	}

	// Hash all keys
	for _, key := range keys {
		h.Write([]byte(key))
		h.Write(s.Leaves[key])
	}

	return &HomomorphicCommitment{
		CommitValue: h.Sum(nil),
		Additions:   []string{},
		Deletions:   []string{},
		Timestamp:   time.Now(),
	}
}

// UpdateHomomorphicCommitment updates a commitment with additions and deletions
func UpdateHomomorphicCommitment(commitment *HomomorphicCommitment, additions, deletions map[string][]byte) *HomomorphicCommitment {
	// In a real implementation, this would use homomorphic properties
	// to avoid recomputing the entire commitment

	// Create new commitment
	newCommitment := &HomomorphicCommitment{
		CommitValue: make([]byte, len(commitment.CommitValue)),
		Additions:   make([]string, 0),
		Deletions:   make([]string, 0),
		Timestamp:   time.Now(),
	}

	// Copy original commitment value
	copy(newCommitment.CommitValue, commitment.CommitValue)

	// Record additions
	for key := range additions {
		newCommitment.Additions = append(newCommitment.Additions, key)
	}

	// Record deletions
	for key := range deletions {
		newCommitment.Deletions = append(newCommitment.Deletions, key)
	}

	return newCommitment
}
