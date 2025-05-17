// pkg/cap/entropy.go
package cap

import (
	"fmt"
	"math"

	"github.com/qasim/blockchain_assignment_3/pkg/utils"
)

// EntropyCalculator calculates information entropy for conflict detection
type EntropyCalculator struct {
	// Cache of entropy values to avoid recalculation
	entropyCache map[string]float64
}

// NewEntropyCalculator creates a new entropy calculator
func NewEntropyCalculator() *EntropyCalculator {
	utils.LogToFile("conflict_entropy", "Initializing new entropy calculator")
	return &EntropyCalculator{
		entropyCache: make(map[string]float64),
	}
}

// CalculateEntropy computes Shannon entropy of the given data
func (e *EntropyCalculator) CalculateEntropy(data []byte) float64 {
	// Generate a cache key
	cacheKey := fmt.Sprintf("%x", data[:min(32, len(data))])

	// Check if we've already calculated this entropy
	if cachedEntropy, exists := e.entropyCache[cacheKey]; exists {
		utils.LogToFile("conflict_entropy", fmt.Sprintf("Using cached entropy value %.4f for data prefix %s",
			cachedEntropy, cacheKey[:8]))
		return cachedEntropy
	}

	// Count frequencies of each byte value
	frequencies := make(map[byte]int)
	for _, b := range data {
		frequencies[b]++
	}

	// Calculate entropy
	dataLength := float64(len(data))
	entropy := 0.0

	for _, count := range frequencies {
		probability := float64(count) / dataLength
		entropy -= probability * math.Log2(probability)
	}

	// Cache the result
	e.entropyCache[cacheKey] = entropy

	utils.LogToFile("conflict_entropy", fmt.Sprintf("Calculated entropy %.4f for data with %d bytes, %d unique values",
		entropy, len(data), len(frequencies)))

	return entropy
}

// CalculateRelativeEntropy computes Kullback-Leibler divergence between two data sets
func (e *EntropyCalculator) CalculateRelativeEntropy(data1, data2 []byte) float64 {
	// Generate histograms
	hist1 := e.generateHistogram(data1)
	hist2 := e.generateHistogram(data2)

	// Make sure all keys in hist1 exist in hist2 (with at least a small probability)
	for key := range hist1 {
		if _, exists := hist2[key]; !exists {
			hist2[key] = 0.00001 // Small non-zero value
		}
	}

	// Calculate KL divergence
	relativeEntropy := 0.0
	for key, prob1 := range hist1 {
		prob2 := hist2[key]
		relativeEntropy += prob1 * math.Log2(prob1/prob2)
	}

	utils.LogToFile("conflict_entropy", fmt.Sprintf("Calculated relative entropy (KL divergence) %.4f between data sets of %d and %d bytes",
		relativeEntropy, len(data1), len(data2)))

	return relativeEntropy
}

// CompareStateEntropy compares two states using various entropy-based methods
func (e *EntropyCalculator) CompareStateEntropy(state1, state2 []byte) (float64, string) {
	// Calculate basic entropy for both states
	entropy1 := e.CalculateEntropy(state1)
	entropy2 := e.CalculateEntropy(state2)

	// Calculate relative entropy in both directions
	kl1to2 := e.CalculateRelativeEntropy(state1, state2)
	kl2to1 := e.CalculateRelativeEntropy(state2, state1)

	// Calculate Jensen-Shannon divergence (symmetric)
	jsDiv := 0.5 * (kl1to2 + kl2to1)

	// Calculate histogram similarity
	histSimilarity := e.calculateHistogramSimilarity(state1, state2)

	// Calculate sequence similarity
	seqSimilarity := e.calculateSequenceSimilarity(state1, state2)

	// Combine metrics to get overall similarity score
	// Lower score means higher similarity
	similarityScore := jsDiv*0.4 + (1-histSimilarity)*0.3 + (1-seqSimilarity)*0.3

	// Determine conflict status
	var resultDescription string
	if similarityScore < 0.1 {
		resultDescription = "highly similar, no conflict"
	} else if similarityScore < 0.3 {
		resultDescription = "similar, minor conflict"
	} else if similarityScore < 0.6 {
		resultDescription = "different, moderate conflict"
	} else {
		resultDescription = "very different, major conflict"
	}

	utils.LogToFile("conflict_entropy", fmt.Sprintf("State comparison: entropy1=%.4f, entropy2=%.4f, similarity=%.4f, JS-div=%.4f, hist-sim=%.4f, seq-sim=%.4f, result: %s",
		entropy1, entropy2, 1-similarityScore, jsDiv, histSimilarity, seqSimilarity, resultDescription))

	return similarityScore, resultDescription
}

// generateHistogram creates a normalized histogram of byte values
func (e *EntropyCalculator) generateHistogram(data []byte) map[byte]float64 {
	// Count frequencies
	counts := make(map[byte]int)
	for _, b := range data {
		counts[b]++
	}

	// Normalize to probabilities
	probabilities := make(map[byte]float64)
	dataLength := float64(len(data))

	for b, count := range counts {
		probabilities[b] = float64(count) / dataLength
	}

	return probabilities
}

// calculateHistogramSimilarity measures how similar the byte distributions are
func (e *EntropyCalculator) calculateHistogramSimilarity(data1, data2 []byte) float64 {
	hist1 := e.generateHistogram(data1)
	hist2 := e.generateHistogram(data2)

	// Get all unique keys
	allKeys := make(map[byte]bool)
	for k := range hist1 {
		allKeys[k] = true
	}
	for k := range hist2 {
		allKeys[k] = true
	}

	// Calculate cosine similarity
	dotProduct := 0.0
	norm1 := 0.0
	norm2 := 0.0

	for key := range allKeys {
		val1 := hist1[key] // Will be 0 if key not present
		val2 := hist2[key] // Will be 0 if key not present

		dotProduct += val1 * val2
		norm1 += val1 * val1
		norm2 += val2 * val2
	}

	// Avoid division by zero
	if norm1 == 0 || norm2 == 0 {
		return 0
	}

	similarity := dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
	return similarity
}

// calculateSequenceSimilarity measures how similar the byte sequences are
func (e *EntropyCalculator) calculateSequenceSimilarity(data1, data2 []byte) float64 {
	// Use a simplified version for performance
	// In a real implementation, use more sophisticated algo like Levenshtein or Smith-Waterman

	minLen := min(len(data1), len(data2))
	maxLen := max(len(data1), len(data2))

	// If either is empty, similarity is 0
	if minLen == 0 {
		return 0
	}

	// Count matching bytes in the same positions
	matches := 0
	for i := 0; i < minLen; i++ {
		if data1[i] == data2[i] {
			matches++
		}
	}

	// Calculate similarity ratio
	similarity := float64(matches) / float64(maxLen)
	return similarity
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
