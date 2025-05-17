// pkg/utils/bloom.go
package utils

import (
	"crypto/sha256"
	"fmt"
	"math"
	"sync"
)

// BloomFilter implements a simple bloom filter for approximate membership queries
type BloomFilter struct {
	bits          []bool
	hashFunctions int
	mutex         sync.RWMutex
}

// NewBloomFilter creates a new bloom filter
func NewBloomFilter(size int, hashFunctions int) *BloomFilter {
	return &BloomFilter{
		bits:          make([]bool, size),
		hashFunctions: hashFunctions,
	}
}

// OptimalBloomFilter creates a bloom filter with optimal parameters
func OptimalBloomFilter(expectedItems int, falsePositiveRate float64) *BloomFilter {
	// Calculate optimal size
	size := int(-float64(expectedItems) * math.Log(falsePositiveRate) / math.Pow(math.Log(2), 2))

	// Calculate optimal number of hash functions
	hashFunctions := int(float64(size) / float64(expectedItems) * math.Log(2))

	// Ensure minimum values
	if size < 1000 {
		size = 1000
	}
	if hashFunctions < 3 {
		hashFunctions = 3
	}

	LogToFile("verification_bloom", fmt.Sprintf("Created bloom filter with size %d bits and %d hash functions (expected items: %d, false positive rate: %.5f)",
		size, hashFunctions, expectedItems, falsePositiveRate))

	return NewBloomFilter(size, hashFunctions)
}

// Add adds an item to the bloom filter
func (bf *BloomFilter) Add(item []byte) {
	bf.mutex.Lock()
	defer bf.mutex.Unlock()

	for i := 0; i < bf.hashFunctions; i++ {
		position := bf.hash(item, i) % len(bf.bits)
		bf.bits[position] = true
	}

	LogToFile("verification_bloom", fmt.Sprintf("Added item to bloom filter (hash prefix: %x)", item[:4]))
}

// Contains checks if an item might be in the set
func (bf *BloomFilter) Contains(item []byte) bool {
	bf.mutex.RLock()
	defer bf.mutex.RUnlock()

	for i := 0; i < bf.hashFunctions; i++ {
		position := bf.hash(item, i) % len(bf.bits)
		if !bf.bits[position] {
			LogToFile("verification_bloom", fmt.Sprintf("Item definitely not in set (hash prefix: %x)", item[:4]))
			return false
		}
	}

	// May be a false positive
	LogToFile("verification_bloom", fmt.Sprintf("Item possibly in set (hash prefix: %x)", item[:4]))
	return true
}

// EstimateFalsePositiveRate estimates the current false positive rate
func (bf *BloomFilter) EstimateFalsePositiveRate(itemCount int) float64 {
	bf.mutex.RLock()
	defer bf.mutex.RUnlock()

	// Count set bits
	setBits := 0
	for _, bit := range bf.bits {
		if bit {
			setBits++
		}
	}

	// Calculate probability of a bit being set
	p := float64(setBits) / float64(len(bf.bits))

	// Calculate false positive rate (probability that all k positions are set)
	fpr := math.Pow(p, float64(bf.hashFunctions))

	LogToFile("verification_bloom", fmt.Sprintf("Estimated false positive rate: %.6f (set bits: %d/%d, items: %d)",
		fpr, setBits, len(bf.bits), itemCount))

	return fpr
}

// hash creates different hash functions by combining the data with a seed
func (bf *BloomFilter) hash(data []byte, seed int) int {
	h := sha256.New()
	h.Write(data)
	h.Write([]byte{byte(seed)})

	sum := h.Sum(nil)
	result := 0
	for i := 0; i < 4; i++ {
		result = (result << 8) | int(sum[i])
	}

	return int(math.Abs(float64(result)))
}
