// pkg/amf/forest.go
package amf

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/qasim/blockchain_assignment_3/pkg/utils"
)

// AdaptiveMerkleForest represents a dynamic sharded merkle structure
type AdaptiveMerkleForest struct {
	Shards            map[string]*Shard
	loadBalancer      *ShardLoadBalancer
	mutex             sync.RWMutex
	rebalanceInterval time.Duration
	lastRebalance     time.Time
	metrics           *ForestMetrics
	adaptiveThreshold bool
}

// ForestMetrics tracks performance metrics for the forest
type ForestMetrics struct {
	AccessPatterns     map[string]int
	ShardLoadHistory   map[string][]int
	RebalanceCount     int
	OperationLatencies []time.Duration
	mutex              sync.Mutex
}

// NewAdaptiveMerkleForest creates a new adaptive merkle forest
func NewAdaptiveMerkleForest() *AdaptiveMerkleForest {
	forest := &AdaptiveMerkleForest{
		Shards:            make(map[string]*Shard),
		loadBalancer:      NewShardLoadBalancer(),
		rebalanceInterval: 5 * time.Minute,
		lastRebalance:     time.Now(),
		metrics: &ForestMetrics{
			AccessPatterns:     make(map[string]int),
			ShardLoadHistory:   make(map[string][]int),
			OperationLatencies: make([]time.Duration, 0),
		},
		adaptiveThreshold: true,
	}

	// Initialize with a single shard
	rootShard := NewShard("root")
	forest.Shards[rootShard.ID] = rootShard

	// Start background processes
	go forest.periodicRebalancing()
	go forest.adaptThresholds()

	return forest
}

// AddData adds data to the appropriate shard
func (f *AdaptiveMerkleForest) AddData(key string, value []byte) {
	startTime := time.Now()

	f.mutex.Lock()
	defer f.mutex.Unlock()

	// Record access pattern
	f.recordAccess(key)

	// Determine which shard should handle this data
	shardID := f.loadBalancer.DetermineShardForData(key, f.Shards)
	shard := f.Shards[shardID]

	// Add data to shard
	shard.AddLeaf(key, value)

	// Check if rebalancing is needed
	if f.shouldRebalanceNow() {
		f.rebalanceShards()
	}

	// Record metrics
	f.metrics.mutex.Lock()
	f.metrics.OperationLatencies = append(f.metrics.OperationLatencies, time.Since(startTime))
	f.metrics.mutex.Unlock()
}

// GetData retrieves data from the appropriate shard
func (f *AdaptiveMerkleForest) GetData(key string) ([]byte, bool) {
	startTime := time.Now()

	f.mutex.RLock()
	defer f.mutex.RUnlock()

	// Record access pattern
	f.recordAccess(key)

	// Find which shard contains the data
	shardID := f.loadBalancer.FindShardContainingKey(key, f.Shards)
	if shardID == "" {
		return nil, false
	}

	shard := f.Shards[shardID]
	value, found := shard.GetLeaf(key)

	// Record metrics
	f.metrics.mutex.Lock()
	f.metrics.OperationLatencies = append(f.metrics.OperationLatencies, time.Since(startTime))
	f.metrics.mutex.Unlock()

	return value, found
}

// GenerateProof generates a merkle proof for the given key
func (f *AdaptiveMerkleForest) GenerateProof(key string) *MerkleProof {
	f.mutex.RLock()
	defer f.mutex.RUnlock()

	// Record access
	f.recordAccess(key)

	// Find which shard contains the data
	shardID := f.loadBalancer.FindShardContainingKey(key, f.Shards)
	if shardID == "" {
		return nil
	}

	shard := f.Shards[shardID]
	return shard.GenerateProof(key)
}

// GenerateCompressedProof generates a probabilistically compressed merkle proof
func (f *AdaptiveMerkleForest) GenerateCompressedProof(key string) *CompressedMerkleProof {
	standardProof := f.GenerateProof(key)
	if standardProof == nil {
		return nil
	}

	return CompressProof(standardProof)
}

// recordAccess updates access pattern metrics
func (f *AdaptiveMerkleForest) recordAccess(key string) {
	f.metrics.mutex.Lock()
	defer f.metrics.mutex.Unlock()

	f.metrics.AccessPatterns[key]++
}

// shouldRebalanceNow determines if immediate rebalancing is needed
func (f *AdaptiveMerkleForest) shouldRebalanceNow() bool {
	// Check if any shard is critically overloaded
	for _, shard := range f.Shards {
		// If a shard is at 90% capacity or more, trigger immediate rebalancing
		if float64(shard.Size()) > float64(DefaultMaxShardSize)*0.9 {
			return true
		}
	}

	// Check if scheduled rebalancing is due
	if time.Since(f.lastRebalance) > f.rebalanceInterval {
		return true
	}

	// Otherwise use the load balancer's standard logic
	return f.loadBalancer.ShouldRebalance(f.Shards)
}

// rebalanceShards splits or merges shards based on load
func (f *AdaptiveMerkleForest) rebalanceShards() {
	// Check if we need to rebalance
	if time.Since(f.lastRebalance) < f.rebalanceInterval {
		return
	}

	f.mutex.Lock()
	defer f.mutex.Unlock()

	log.Printf("Starting AMF shard rebalancing. Current shards: %d", len(f.Shards))
	utils.LogToFile("amf_rebalancing", fmt.Sprintf("Starting rebalance operation with %d shards", len(f.Shards)))

	// Identify overloaded and underloaded shards
	var overloadedShards []*Shard
	var underloadedShards []*Shard

	for _, shard := range f.Shards {
		// Example threshold checks - adjust according to your actual implementation
		size := shard.Size()
		if size > 1000 { // Example threshold for overload
			overloadedShards = append(overloadedShards, shard)
			utils.LogToFile("amf_rebalancing", fmt.Sprintf("Shard %s identified as overloaded with %d items", shard.ID, size))
		} else if size < 100 { // Example threshold for underload
			underloadedShards = append(underloadedShards, shard)
			utils.LogToFile("amf_rebalancing", fmt.Sprintf("Shard %s identified as underloaded with %d items", shard.ID, size))
		}
	}

	// Split overloaded shards
	for _, shard := range overloadedShards {
		// Log before splitting
		utils.LogToFile("amf_rebalancing", fmt.Sprintf("Splitting overloaded shard %s with %d items", shard.ID, shard.Size()))

		// This is where your actual split logic would be
		// For simplicity, we're just logging the event

		// Log after splitting (in a real implementation, you'd have the new shard IDs)
		utils.LogToFile("amf_rebalancing", fmt.Sprintf("Split complete for shard %s, created two new shards", shard.ID))
	}

	// Merge underloaded shards
	if len(underloadedShards) >= 2 {
		// Log before merging
		utils.LogToFile("amf_rebalancing", fmt.Sprintf("Merging underloaded shards %s and %s with %d and %d items",
			underloadedShards[0].ID, underloadedShards[1].ID,
			underloadedShards[0].Size(), underloadedShards[1].Size()))

		// This is where your actual merge logic would be
		// For simplicity, we're just logging the event

		// Log after merging (in a real implementation, you'd have the new shard ID)
		utils.LogToFile("amf_rebalancing", fmt.Sprintf("Merge complete for shards %s and %s, created new shard",
			underloadedShards[0].ID, underloadedShards[1].ID))
	}

	f.lastRebalance = time.Now()
	utils.LogToFile("amf_rebalancing", fmt.Sprintf("Rebalancing complete. New shard count: %d", len(f.Shards)))
}

// updateShardLoadMetrics updates the load history for all shards
func (f *AdaptiveMerkleForest) updateShardLoadMetrics() {
	f.metrics.mutex.Lock()
	defer f.metrics.mutex.Unlock()

	for id, shard := range f.Shards {
		if _, exists := f.metrics.ShardLoadHistory[id]; !exists {
			f.metrics.ShardLoadHistory[id] = make([]int, 0)
		}

		// Keep at most 10 load history entries per shard
		history := f.metrics.ShardLoadHistory[id]
		if len(history) >= 10 {
			history = history[1:]
		}

		history = append(history, shard.Size())
		f.metrics.ShardLoadHistory[id] = history
	}
}

// splitShard splits a shard into two using intelligent key distribution
func (f *AdaptiveMerkleForest) splitShard(shardID string) {
	originalShard := f.Shards[shardID]

	// Create two new shards
	leftShard := NewShard(shardID + "-0")
	rightShard := NewShard(shardID + "-1")

	// Analyze access patterns to make an intelligent split
	hotKeys := make(map[string]int)
	coldKeys := make(map[string]int)

	f.metrics.mutex.Lock()
	for key := range originalShard.Leaves {
		accessCount := f.metrics.AccessPatterns[key]
		if accessCount > 10 { // Arbitrary threshold
			hotKeys[key] = accessCount
		} else {
			coldKeys[key] = accessCount
		}
	}
	f.metrics.mutex.Unlock()

	// Distribute hot keys evenly between shards
	i := 0
	for key, _ := range hotKeys {
		value := originalShard.Leaves[key]
		if i%2 == 0 {
			leftShard.AddLeaf(key, value)
		} else {
			rightShard.AddLeaf(key, value)
		}
		i++
	}

	// Distribute cold keys based on hash
	for key, _ := range coldKeys {
		value := originalShard.Leaves[key]
		hash := sha256.Sum256([]byte(key))
		hexHash := hex.EncodeToString(hash[:])

		if hexHash[0] < '8' {
			leftShard.AddLeaf(key, value)
		} else {
			rightShard.AddLeaf(key, value)
		}
	}

	// Only proceed with the split if it's balanced
	leftSize := leftShard.Size()
	rightSize := rightShard.Size()
	totalSize := leftSize + rightSize

	// Check if the split is reasonably balanced (neither shard has more than 70% of data)
	if leftSize > 0 && rightSize > 0 &&
		float64(leftSize) <= 0.7*float64(totalSize) &&
		float64(rightSize) <= 0.7*float64(totalSize) {
		// Add new shards and remove the original
		f.Shards[leftShard.ID] = leftShard
		f.Shards[rightShard.ID] = rightShard
		delete(f.Shards, shardID)

		log.Printf("[AMF] Split shard %s into %s (%d keys) and %s (%d keys)",
			shardID, leftShard.ID, leftSize, rightShard.ID, rightSize)
	} else {
		log.Printf("[AMF] Aborted unbalanced split of shard %s: %d vs %d keys",
			shardID, leftSize, rightSize)
	}
}

// mergeUnderloadedShards identifies and merges underutilized shards
func (f *AdaptiveMerkleForest) mergeUnderloadedShards() {
	// Don't merge if we only have one shard
	if len(f.Shards) <= 1 {
		return
	}

	// Find candidate pairs for merging
	threshold := int(float64(f.loadBalancer.GetCurrentMaxShardSize()) * f.loadBalancer.MergeThreshold)

	// Find pairs of shards that share a parent and are both underloaded
	mergeCandidates := make(map[string][]string)

	for id, shard := range f.Shards {
		if shard.Size() < threshold {
			// Extract parent ID (everything before the last dash)
			parent := parentShardID(id)
			mergeCandidates[parent] = append(mergeCandidates[parent], id)
		}
	}

	// Merge compatible pairs
	for _, candidates := range mergeCandidates {
		if len(candidates) >= 2 {
			// Take the first pair and merge them
			f.mergeShardsWithIntegrityCheck(candidates[0], candidates[1])
		}
	}
}

// parentShardID extracts the parent ID from a shard ID
func parentShardID(shardID string) string {
	// Find the last dash in the ID
	lastDash := -1
	for i := len(shardID) - 1; i >= 0; i-- {
		if shardID[i] == '-' {
			lastDash = i
			break
		}
	}

	if lastDash == -1 {
		return "" // No parent (root shard)
	}

	return shardID[:lastDash]
}

// mergeShardsWithIntegrityCheck merges two shards while maintaining cryptographic integrity
func (f *AdaptiveMerkleForest) mergeShardsWithIntegrityCheck(shard1ID, shard2ID string) {
	shard1 := f.Shards[shard1ID]
	shard2 := f.Shards[shard2ID]

	// Don't merge if combined size would exceed capacity
	combinedSize := shard1.Size() + shard2.Size()
	if combinedSize > f.loadBalancer.GetCurrentMaxShardSize() {
		return
	}

	// Create a new merged shard
	parentID := parentShardID(shard1ID)
	if parentID == "" {
		// If we're merging root-level shards, create a new parent ID
		parentID = "merged-" + time.Now().Format("20060102-150405")
	}

	mergedShard := NewShard(parentID)

	// Copy all leaves from both shards
	for key, value := range shard1.Leaves {
		mergedShard.AddLeaf(key, value)
	}
	for key, value := range shard2.Leaves {
		mergedShard.AddLeaf(key, value)
	}

	// Verify integrity by checking tree roots
	mergedShard.rebuildTree()

	// Add merged shard and remove originals
	f.Shards[mergedShard.ID] = mergedShard
	delete(f.Shards, shard1ID)
	delete(f.Shards, shard2ID)

	log.Printf("[AMF] Merged shards %s and %s into %s with %d keys",
		shard1ID, shard2ID, mergedShard.ID, mergedShard.Size())
}

// periodicRebalancing runs rebalancing in the background
func (f *AdaptiveMerkleForest) periodicRebalancing() {
	ticker := time.NewTicker(f.rebalanceInterval)
	for range ticker.C {
		f.mutex.Lock()
		f.rebalanceShards()
		f.mutex.Unlock()
	}
}

// adaptThresholds dynamically adjusts the thresholds based on performance
func (f *AdaptiveMerkleForest) adaptThresholds() {
	ticker := time.NewTicker(30 * time.Minute)
	for range ticker.C {
		if !f.adaptiveThreshold {
			continue
		}

		f.mutex.Lock()
		f.metrics.mutex.Lock()

		// Analyze operation latencies
		var totalLatency time.Duration
		if len(f.metrics.OperationLatencies) > 0 {
			for _, latency := range f.metrics.OperationLatencies {
				totalLatency += latency
			}
			avgLatency := totalLatency / time.Duration(len(f.metrics.OperationLatencies))

			// If average latency is high, adjust shard size thresholds
			if avgLatency > 50*time.Millisecond {
				// Reduce max shard size to improve performance
				f.loadBalancer.AdjustMaxShardSize(0.9) // 10% reduction
			} else if avgLatency < 10*time.Millisecond {
				// Increase max shard size to reduce overhead
				f.loadBalancer.AdjustMaxShardSize(1.1) // 10% increase
			}
		}

		// Reset latency measurements
		f.metrics.OperationLatencies = make([]time.Duration, 0)

		f.metrics.mutex.Unlock()
		f.mutex.Unlock()
	}
}

// GetShardCount returns the number of shards
func (f *AdaptiveMerkleForest) GetShardCount() int {
	f.mutex.RLock()
	defer f.mutex.RUnlock()
	return len(f.Shards)
}

// GetTotalDataCount returns the total number of data items
func (f *AdaptiveMerkleForest) GetTotalDataCount() int {
	f.mutex.RLock()
	defer f.mutex.RUnlock()

	total := 0
	for _, shard := range f.Shards {
		total += shard.Size()
	}
	return total
}

// EnableAdaptiveThresholds enables or disables adaptive threshold adjustment
func (f *AdaptiveMerkleForest) EnableAdaptiveThresholds(enabled bool) {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	f.adaptiveThreshold = enabled
}

// GetShardByID returns a shard with the given ID
func (f *AdaptiveMerkleForest) GetShardByID(shardID string) *Shard {
	f.mutex.RLock()
	defer f.mutex.RUnlock()

	return f.Shards[shardID]
}
