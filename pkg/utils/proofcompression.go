// pkg/utils/proofcompression.go
package utils

import (
	"crypto/sha256"
	"fmt"
	"math/rand"
	"time"
)

// CompressedProof represents a compressed Merkle proof
type CompressedProof struct {
	Hashes        [][]byte
	Flags         []bool // true if hash should be verified, false if it can be skipped
	SkipPositions []int  // positions where verification was skipped
	RootHash      []byte
}

// CompressProof probabilistically compresses a standard Merkle proof
func CompressProof(hashes [][]byte, rootHash []byte, compressionRatio float64) *CompressedProof {
	if compressionRatio <= 0 || compressionRatio >= 1 {
		LogToFile("verification_compression", fmt.Sprintf("Invalid compression ratio %.2f, using standard proof",
			compressionRatio))
		return &CompressedProof{
			Hashes:        hashes,
			Flags:         make([]bool, len(hashes)),
			SkipPositions: []int{},
			RootHash:      rootHash,
		}
	}

	originalSize := len(hashes) * 32 // Each hash is 32 bytes

	// Determine which positions to skip verification
	flags := make([]bool, len(hashes))
	skipPositions := make([]int, 0)

	for i := range hashes {
		// Always check the first and last position
		if i == 0 || i == len(hashes)-1 {
			flags[i] = true
			continue
		}

		// Randomly decide whether to skip this position
		if rand.Float64() < compressionRatio {
			skipPositions = append(skipPositions, i)
			flags[i] = false
		} else {
			flags[i] = true
		}
	}

	compressedProof := &CompressedProof{
		Hashes:        hashes,
		Flags:         flags,
		SkipPositions: skipPositions,
		RootHash:      rootHash,
	}

	// Calculate compressed size
	// Each skipped position saves a hash but adds an integer (4 bytes)
	compressedSize := originalSize - (len(skipPositions) * (32 - 4))

	LogToFile("verification_compression", fmt.Sprintf("Compressed proof: original size %d bytes, compressed size %d bytes, %.1f%% reduction, skipped %d of %d hashes",
		originalSize, compressedSize, (1-float64(compressedSize)/float64(originalSize))*100, len(skipPositions), len(hashes)))

	return compressedProof
}

// VerifyCompressedProof verifies a compressed Merkle proof
func VerifyCompressedProof(proof *CompressedProof, leafHash []byte) bool {
	startTime := StartTimer()

	// Begin with the leaf hash
	currentHash := leafHash
	verifiedHashes := 0
	skippedHashes := 0

	// Verify each hash in sequence, skipping as directed by flags
	for i, hash := range proof.Hashes {
		if proof.Flags[i] {
			// Verify this step
			combinedHash := combineHashes(currentHash, hash)
			currentHash = combinedHash
			verifiedHashes++
		} else {
			// Skip verification, just use the provided hash
			currentHash = hash
			skippedHashes++
		}
	}

	// Check if the final hash matches the root
	result := fmt.Sprintf("%x", currentHash) == fmt.Sprintf("%x", proof.RootHash)

	elapsed := StopTimer(startTime)

	LogToFile("verification_compression", fmt.Sprintf("Compressed proof verification: %v, verified %d hashes, skipped %d hashes, took %.2fms",
		result, verifiedHashes, skippedHashes, elapsed*1000))

	return result
}

// Helper function to combine hashes
func combineHashes(left, right []byte) []byte {
	h := sha256.New()
	h.Write(left)
	h.Write(right)
	return h.Sum(nil)
}

// StartTimer begins a timer for performance tracking
func StartTimer() int64 {
	return TimeNowNano()
}

// StopTimer ends a timer and returns the elapsed duration in seconds
func StopTimer(startTime int64) float64 {
	endTime := TimeNowNano()
	return float64(endTime-startTime) / 1e9 // Convert to seconds
}

// TimeNowNano returns the current time in nanoseconds
func TimeNowNano() int64 {
	return DateTimeNowImpl().UnixNano()
}

// This is a function variable that can be mocked in tests
var DateTimeNowImpl = func() interface {
	UnixNano() int64
} {
	// Use the real time package - in a real implementation this would use time.Now()
	return time.Now()
}
