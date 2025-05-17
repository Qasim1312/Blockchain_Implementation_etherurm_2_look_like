// pkg/cap/conflict.go
package cap

import (
	"crypto/sha256"
	"encoding/hex"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// VectorClock represents a vector clock for causal consistency
type VectorClock map[string]uint64

// ConflictManager handles conflict detection and resolution
type ConflictManager struct {
	nodeID               string
	vectorClock          VectorClock
	lastSeen             map[string]VectorClock
	entropyCache         map[string]float64
	conflictHistory      []Conflict
	resolutionStrategies map[string]ConflictResolutionStrategy
	mutex                sync.RWMutex
	entropyThreshold     float64
	histogramCache       map[string]map[byte]int
}

// Conflict represents a detected conflict
type Conflict struct {
	ID              string
	State1Hash      string
	State2Hash      string
	EntropyDiff     float64
	Timestamp       time.Time
	Resolution      string
	ResolutionState string
	Probability     float64
}

// ConflictResolutionStrategy defines a method for resolving conflicts
type ConflictResolutionStrategy struct {
	Name                    string
	Priority                int
	StatePreference         string // "higher_entropy", "lower_entropy", "newer", "older"
	ProbabilisticResolution bool
}

// NewConflictManager creates a new conflict manager
func NewConflictManager(nodeID string) *ConflictManager {
	rand.Seed(time.Now().UnixNano())

	return &ConflictManager{
		nodeID:          nodeID,
		vectorClock:     make(VectorClock),
		lastSeen:        make(map[string]VectorClock),
		entropyCache:    make(map[string]float64),
		conflictHistory: make([]Conflict, 0),
		resolutionStrategies: map[string]ConflictResolutionStrategy{
			"information_theoretic": {
				Name:                    "information_theoretic",
				Priority:                10,
				StatePreference:         "higher_entropy",
				ProbabilisticResolution: false,
			},
			"probabilistic": {
				Name:                    "probabilistic",
				Priority:                5,
				StatePreference:         "higher_entropy",
				ProbabilisticResolution: true,
			},
			"timestamp": {
				Name:                    "timestamp",
				Priority:                3,
				StatePreference:         "newer",
				ProbabilisticResolution: false,
			},
		},
		mutex:            sync.RWMutex{},
		entropyThreshold: 0.3,
		histogramCache:   make(map[string]map[byte]int),
	}
}

// IncrementClock increments this node's vector clock
func (cm *ConflictManager) IncrementClock() {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	cm.vectorClock[cm.nodeID]++
}

// MergeClock merges the incoming vector clock with our own
func (cm *ConflictManager) MergeClock(other VectorClock) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	for node, clock := range other {
		if cm.vectorClock[node] < clock {
			cm.vectorClock[node] = clock
		}
	}
}

// GetVectorClock returns a copy of the current vector clock
func (cm *ConflictManager) GetVectorClock() VectorClock {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()

	return copyVectorClock(cm.vectorClock)
}

// CheckCausalOrder checks if an event respects causal ordering
func (cm *ConflictManager) CheckCausalOrder(sender string, clock VectorClock) bool {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	// Store the last seen clock for this sender
	if _, exists := cm.lastSeen[sender]; !exists {
		cm.lastSeen[sender] = make(VectorClock)
	}

	// Check if this message is newer than the last one from this sender
	if clock[sender] <= cm.lastSeen[sender][sender] {
		return false // Message is older or a duplicate
	}

	// Check if it respects causal ordering
	for node, val := range clock {
		if node != sender && val > cm.vectorClock[node] {
			return false // Message depends on events we haven't seen
		}
	}

	// Update last seen clock for this sender
	cm.lastSeen[sender] = copyVectorClock(clock)

	return true
}

// DetectConflict detects if two states conflict
func (cm *ConflictManager) DetectConflict(state1, state2 []byte) (bool, float64) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	// Generate hashes for states
	hash1 := sha256.Sum256(state1)
	hash2 := sha256.Sum256(state2)

	hashStr1 := hex.EncodeToString(hash1[:])
	hashStr2 := hex.EncodeToString(hash2[:])

	// If identical hashes, no conflict
	if hashStr1 == hashStr2 {
		return false, 0.0
	}

	// Use multiple detection methods for higher confidence

	// 1. Entropy-based detection
	entropy1 := cm.calculateEntropyWithCache(state1, hashStr1)
	entropy2 := cm.calculateEntropyWithCache(state2, hashStr2)
	entropyDiff := math.Abs(entropy1 - entropy2)

	// 2. Histogram comparison (content-aware)
	histogramSimilarity := cm.calculateHistogramSimilarity(state1, state2, hashStr1, hashStr2)

	// 3. Sequence similarity (look for sections that are identical)
	sequenceScore := cm.calculateSequenceSimilarity(state1, state2)

	// Combine these metrics with weighted averaging
	conflictScore := (entropyDiff * 0.4) + ((1.0 - histogramSimilarity) * 0.3) + ((1.0 - sequenceScore) * 0.3)

	// States are considered in conflict if the combined score exceeds the threshold
	isConflict := conflictScore > cm.entropyThreshold

	// Record conflict if detected
	if isConflict {
		conflict := Conflict{
			ID:          hashStr1[:8] + "-" + hashStr2[:8],
			State1Hash:  hashStr1,
			State2Hash:  hashStr2,
			EntropyDiff: entropyDiff,
			Timestamp:   time.Now(),
		}
		cm.conflictHistory = append(cm.conflictHistory, conflict)

		// Limit history size
		if len(cm.conflictHistory) > 100 {
			cm.conflictHistory = cm.conflictHistory[1:]
		}
	}

	return isConflict, conflictScore
}

// calculateHistogramSimilarity compares byte frequency histograms
func (cm *ConflictManager) calculateHistogramSimilarity(data1, data2 []byte, hash1, hash2 string) float64 {
	// Get or create histograms
	hist1 := cm.getHistogram(data1, hash1)
	hist2 := cm.getHistogram(data2, hash2)

	// Compare histograms using cosine similarity
	var dotProduct float64
	var norm1, norm2 float64

	// Calculate dot product and vector norms
	for b := 0; b < 256; b++ {
		byte_val := byte(b)
		v1 := float64(hist1[byte_val])
		v2 := float64(hist2[byte_val])

		dotProduct += v1 * v2
		norm1 += v1 * v1
		norm2 += v2 * v2
	}

	norm1 = math.Sqrt(norm1)
	norm2 = math.Sqrt(norm2)

	// Avoid division by zero
	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}

	// Cosine similarity (1.0 = identical, 0.0 = completely different)
	return dotProduct / (norm1 * norm2)
}

// getHistogram gets or calculates byte histogram for data
func (cm *ConflictManager) getHistogram(data []byte, hash string) map[byte]int {
	if hist, exists := cm.histogramCache[hash]; exists {
		return hist
	}

	// Create histogram
	hist := make(map[byte]int)
	for _, b := range data {
		hist[b]++
	}

	// Cache for future use
	cm.histogramCache[hash] = hist

	return hist
}

// calculateSequenceSimilarity calculates similarity based on common sequences
func (cm *ConflictManager) calculateSequenceSimilarity(data1, data2 []byte) float64 {
	// This is a simplified implementation of sequence similarity
	// A real implementation would use more sophisticated algorithms like
	// longest common subsequence or Smith-Waterman

	// We'll use a simple approach with sliding windows
	windowSize := 8
	if len(data1) < windowSize || len(data2) < windowSize {
		windowSize = int(math.Min(float64(len(data1)), float64(len(data2))))
	}

	if windowSize < 2 {
		return 0.0 // Too small for meaningful comparison
	}

	// Create window hashes for first array
	windows1 := make(map[string]bool)
	for i := 0; i <= len(data1)-windowSize; i++ {
		window := data1[i : i+windowSize]
		windowHash := string(window) // Use string as simple hash
		windows1[windowHash] = true
	}

	// Count matches in second array
	matches := 0
	for i := 0; i <= len(data2)-windowSize; i++ {
		window := data2[i : i+windowSize]
		windowHash := string(window)
		if windows1[windowHash] {
			matches++
		}
	}

	// Calculate similarity ratio
	possibleWindows := len(data1) - windowSize + 1
	if possibleWindows <= 0 {
		return 0.0
	}

	return float64(matches) / float64(possibleWindows)
}

// calculateEntropyWithCache calculates Shannon entropy of the data with caching
func (cm *ConflictManager) calculateEntropyWithCache(data []byte, hashStr string) float64 {
	// Check cache first
	if entropy, found := cm.entropyCache[hashStr]; found {
		return entropy
	}

	// Count occurrences of each byte
	counts := make(map[byte]int)
	for _, b := range data {
		counts[b]++
	}

	// Calculate entropy
	entropy := 0.0
	length := float64(len(data))

	for _, count := range counts {
		p := float64(count) / length
		entropy -= p * logBase2(p)
	}

	// Cache result
	cm.entropyCache[hashStr] = entropy
	return entropy
}

// logBase2 calculates log base 2
func logBase2(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return math.Log2(x)
}

// ResolveConflict resolves conflicts between two states
func (cm *ConflictManager) ResolveConflict(state1, state2 []byte, timestamp1, timestamp2 time.Time) []byte {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	// Get state hashes
	hash1 := sha256.Sum256(state1)
	hash2 := sha256.Sum256(state2)
	hashStr1 := hex.EncodeToString(hash1[:])
	hashStr2 := hex.EncodeToString(hash2[:])

	// Sort strategies by priority (highest first)
	var strategies []ConflictResolutionStrategy
	for _, strategy := range cm.resolutionStrategies {
		strategies = append(strategies, strategy)
	}
	sort.Slice(strategies, func(i, j int) bool {
		return strategies[i].Priority > strategies[j].Priority
	})

	// Apply strategies in order until one resolves the conflict
	for _, strategy := range strategies {
		var resolvedState []byte
		var probability float64 = 1.0

		switch strategy.Name {
		case "information_theoretic":
			entropy1 := cm.calculateEntropyWithCache(state1, hashStr1)
			entropy2 := cm.calculateEntropyWithCache(state2, hashStr2)

			if strategy.StatePreference == "higher_entropy" {
				if entropy1 >= entropy2 {
					resolvedState = state1
				} else {
					resolvedState = state2
				}
			} else {
				// Lower entropy preference
				if entropy1 <= entropy2 {
					resolvedState = state1
				} else {
					resolvedState = state2
				}
			}

		case "probabilistic":
			entropy1 := cm.calculateEntropyWithCache(state1, hashStr1)
			entropy2 := cm.calculateEntropyWithCache(state2, hashStr2)

			// Calculate probabilities based on relative entropy
			totalEntropy := entropy1 + entropy2
			if totalEntropy > 0 {
				prob1 := entropy1 / totalEntropy

				// Use probability to choose
				if rand.Float64() < prob1 {
					resolvedState = state1
					probability = prob1
				} else {
					resolvedState = state2
					probability = 1.0 - prob1
				}
			} else {
				// Equal or zero entropy, pick randomly
				if rand.Float64() < 0.5 {
					resolvedState = state1
				} else {
					resolvedState = state2
				}
				probability = 0.5
			}

		case "timestamp":
			if strategy.StatePreference == "newer" {
				if timestamp1.After(timestamp2) {
					resolvedState = state1
				} else {
					resolvedState = state2
				}
			} else {
				// Older preference
				if timestamp1.Before(timestamp2) {
					resolvedState = state1
				} else {
					resolvedState = state2
				}
			}
		}

		// If strategy resolved the conflict, record and return
		if resolvedState != nil {
			// Find or create conflict record
			conflictID := hashStr1[:8] + "-" + hashStr2[:8]
			var conflict *Conflict

			// Look for existing conflict
			for i := range cm.conflictHistory {
				if cm.conflictHistory[i].ID == conflictID {
					conflict = &cm.conflictHistory[i]
					break
				}
			}

			// Record resolution
			if conflict != nil {
				hashBytes := sha256.Sum256(resolvedState)
				resolvedHash := hex.EncodeToString(hashBytes[:])
				conflict.Resolution = strategy.Name
				conflict.ResolutionState = resolvedHash
				conflict.Probability = probability
			}

			log.Printf("[CAP] Conflict %s resolved using strategy %s with probability %.2f",
				conflictID, strategy.Name, probability)

			return resolvedState
		}
	}

	// Default: return first state if no strategy resolved the conflict
	return state1
}

// GetConflictHistory returns the history of conflicts
func (cm *ConflictManager) GetConflictHistory() []Conflict {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()

	return cm.conflictHistory
}

// AddResolutionStrategy adds a custom conflict resolution strategy
func (cm *ConflictManager) AddResolutionStrategy(name string, priority int, statePreference string, probabilistic bool) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	cm.resolutionStrategies[name] = ConflictResolutionStrategy{
		Name:                    name,
		Priority:                priority,
		StatePreference:         statePreference,
		ProbabilisticResolution: probabilistic,
	}
}

// SetEntropyThreshold sets the threshold for conflict detection
func (cm *ConflictManager) SetEntropyThreshold(threshold float64) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	cm.entropyThreshold = threshold
}

// copyVectorClock creates a copy of a vector clock
func copyVectorClock(clock VectorClock) VectorClock {
	result := make(VectorClock)
	for k, v := range clock {
		result[k] = v
	}
	return result
}
