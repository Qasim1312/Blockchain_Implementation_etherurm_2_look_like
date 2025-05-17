// pkg/bft/leader.go
package bft

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strconv"
	"sync"

	"github.com/qasim/blockchain_assignment_3/pkg/utils"
)

// VRFLeaderElection implements VRF-based leader selection
type VRFLeaderElection struct {
	nodeSecrets map[string][]byte // Maps node IDs to secret values
	mutex       sync.RWMutex
}

// NewVRFLeaderElection creates a new VRF-based leader election system
func NewVRFLeaderElection() *VRFLeaderElection {
	return &VRFLeaderElection{
		nodeSecrets: make(map[string][]byte),
	}
}

// RegisterNode registers a node with its public key for VRF calculations
// In a real implementation, public keys would be used for cryptographic operations
func (le *VRFLeaderElection) RegisterNode(nodeID string, publicKey []byte) {
	le.mutex.Lock()
	defer le.mutex.Unlock()

	// In a real implementation, you might store the actual public key
	// For this simulation, we'll use the public key as the secret
	le.nodeSecrets[nodeID] = publicKey
}

// GenerateVRFValue generates a VRF output for a particular round and node
func (le *VRFLeaderElection) GenerateVRFValue(roundNumber uint64, nodeID string) ([]byte, error) {
	le.mutex.RLock()
	defer le.mutex.RUnlock()

	// Check if we know this node
	secretKey, exists := le.nodeSecrets[nodeID]
	if !exists {
		return nil, fmt.Errorf("unknown node: %s", nodeID)
	}

	// Prepare input data (round number + node ID)
	input := make([]byte, 8)
	binary.BigEndian.PutUint64(input, roundNumber)
	input = append(input, []byte(nodeID)...)

	// Create HMAC with secret key (simplified VRF)
	// In a real implementation, this would use an actual VRF algorithm
	h := hmac.New(sha256.New, secretKey)
	h.Write(input)
	output := h.Sum(nil)

	return output, nil
}

// VerifyVRFValue verifies a VRF output for a given round and node
func (le *VRFLeaderElection) VerifyVRFValue(roundNumber uint64, nodeID string, vrfValue []byte) bool {
	le.mutex.RLock()
	defer le.mutex.RUnlock()

	// Check if we know this node
	secretKey, exists := le.nodeSecrets[nodeID]
	if !exists {
		return false
	}

	// Generate the expected value
	input := make([]byte, 8)
	binary.BigEndian.PutUint64(input, roundNumber)
	input = append(input, []byte(nodeID)...)

	// Create HMAC with the node's secret key
	h := hmac.New(sha256.New, secretKey)
	h.Write(input)
	expectedValue := h.Sum(nil)

	// Compare the generated value with the provided one
	return bytes.Equal(vrfValue, expectedValue)
}

// DetermineLeader selects a leader based on VRF outputs from all nodes
func (le *VRFLeaderElection) DetermineLeader(round uint64, candidates map[string][]byte) (string, error) {
	if len(candidates) == 0 {
		return "", fmt.Errorf("no candidate values provided")
	}

	// Find the lowest hash value (treated as a uint64)
	var lowestValue uint64 = ^uint64(0) // max uint64
	var leader string

	for nodeID, value := range candidates {
		// Take the first 8 bytes of the hash and interpret as uint64
		if len(value) < 8 {
			continue // Invalid value
		}

		candidateValue := binary.BigEndian.Uint64(value[:8])
		if candidateValue < lowestValue {
			lowestValue = candidateValue
			leader = nodeID
		}
	}

	if leader == "" {
		return "", fmt.Errorf("failed to determine leader")
	}

	return leader, nil
}

// SimulateElection simulates a complete leader election round
func (le *VRFLeaderElection) SimulateElection(round uint64, nodeIDs []string) (string, error) {
	candidates := make(map[string][]byte)

	// Generate candidate values for all nodes
	for _, nodeID := range nodeIDs {
		value, err := le.GenerateVRFValue(round, nodeID)
		if err != nil {
			return "", fmt.Errorf("error generating candidate value for %s: %v", nodeID, err)
		}
		candidates[nodeID] = value
	}

	// Determine leader
	return le.DetermineLeader(round, candidates)
}

// TestVRFImplementation runs a comprehensive test of the VRF implementation
// and logs the results for verification and analysis
func (le *VRFLeaderElection) TestVRFImplementation(rounds int, nodeCount int) error {
	startTime := utils.GetTimestampNano()

	utils.LogToFile("vrf_test", fmt.Sprintf("Starting VRF implementation test with %d rounds and %d nodes", rounds, nodeCount))

	// Generate test nodes
	nodeIDs := make([]string, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodeIDs[i] = fmt.Sprintf("node-%d", i)

		// Generate a random public key for each node
		publicKey := make([]byte, 32)
		_, err := rand.Read(publicKey)
		if err != nil {
			errMsg := fmt.Sprintf("Failed to generate public key for node %s: %v", nodeIDs[i], err)
			utils.LogToFile("vrf_test", errMsg)
			return fmt.Errorf(errMsg)
		}

		le.RegisterNode(nodeIDs[i], publicKey)
		utils.LogToFile("vrf_test", fmt.Sprintf("Registered node %s with public key %s", nodeIDs[i], hex.EncodeToString(publicKey)))
	}

	// Statistics to track
	leaderDistribution := make(map[string]int)
	totalProofGenerationTime := int64(0)
	totalProofVerificationTime := int64(0)
	successfulVerifications := 0
	failedVerifications := 0

	// Run multiple election rounds
	for round := 0; round < rounds; round++ {
		roundID := uint64(round)
		roundStart := utils.GetTimestampNano()

		utils.LogToFile("vrf_test", fmt.Sprintf("--- Starting round %d ---", round))

		// Generate and verify VRF values for each node
		vrfValues := make(map[string][]byte)
		vrfProofs := make(map[string][]byte)

		for _, nodeID := range nodeIDs {
			// Time VRF value and proof generation
			valueStart := utils.GetTimestampNano()
			value, err := le.GenerateVRFValue(roundID, nodeID)
			valueEnd := utils.GetTimestampNano()
			if err != nil {
				errMsg := fmt.Sprintf("Error generating VRF value for node %s in round %d: %v", nodeID, round, err)
				utils.LogToFile("vrf_test", errMsg)
				continue
			}

			proofStart := utils.GetTimestampNano()
			proof, err := le.GenerateVRFProof(roundID, nodeID)
			proofEnd := utils.GetTimestampNano()

			if err != nil {
				errMsg := fmt.Sprintf("Error generating VRF proof for node %s in round %d: %v", nodeID, round, err)
				utils.LogToFile("vrf_test", errMsg)
				continue
			}

			valueGenerationTime := valueEnd - valueStart
			proofGenerationTime := proofEnd - proofStart
			totalProofGenerationTime += proofGenerationTime

			vrfValues[nodeID] = value
			vrfProofs[nodeID] = proof

			// Log the value generation
			metadata := map[string]interface{}{
				"node_id":               nodeID,
				"round":                 round,
				"value_hex":             hex.EncodeToString(value[:8]) + "...",
				"proof_hex":             hex.EncodeToString(proof[:8]) + "...",
				"value_generation_time": utils.FormatDuration(valueGenerationTime),
				"proof_generation_time": utils.FormatDuration(proofGenerationTime),
			}
			utils.LogWithMetadata("vrf_test", "Generated VRF value and proof", metadata)

			// Verify the VRF value
			verifyStart := utils.GetTimestampNano()
			isValid := le.VerifyVRFValue(roundID, nodeID, value)
			verifyEnd := utils.GetTimestampNano()

			verificationTime := verifyEnd - verifyStart
			totalProofVerificationTime += verificationTime

			if isValid {
				successfulVerifications++
				utils.LogToFile("vrf_test", fmt.Sprintf("✓ VRF verification succeeded for node %s in round %d (took %s)",
					nodeID, round, utils.FormatDuration(verificationTime)))
			} else {
				failedVerifications++
				utils.LogToFile("vrf_test", fmt.Sprintf("✗ VRF verification FAILED for node %s in round %d (took %s)",
					nodeID, round, utils.FormatDuration(verificationTime)))
			}
		}

		// Determine leader for this round
		leader, err := le.DetermineLeader(roundID, vrfValues)
		if err != nil {
			errMsg := fmt.Sprintf("Error determining leader for round %d: %v", round, err)
			utils.LogToFile("vrf_test", errMsg)
			continue
		}

		// Update statistics
		leaderDistribution[leader]++

		// Calculate lowest VRF value for logging
		var lowestValue uint64 = ^uint64(0)
		for _, value := range vrfValues {
			if len(value) >= 8 {
				candidateValue := binary.BigEndian.Uint64(value[:8])
				if candidateValue < lowestValue {
					lowestValue = candidateValue
				}
			}
		}

		// Log the leader selection
		metadata := map[string]interface{}{
			"leader":       leader,
			"round":        round,
			"lowest_value": fmt.Sprintf("%016x", lowestValue),
			"candidates":   len(vrfValues),
			"round_time":   utils.FormatDuration(utils.GetTimestampNano() - roundStart),
		}
		utils.LogWithMetadata("vrf_test", "Leader selected for round", metadata)
	}

	// Log final statistics
	totalTime := utils.GetTimestampNano() - startTime
	avgProofGenerationTime := float64(totalProofGenerationTime) / float64(successfulVerifications+failedVerifications)
	avgProofVerificationTime := float64(totalProofVerificationTime) / float64(successfulVerifications+failedVerifications)

	// Log leader distribution fairness
	utils.LogToFile("vrf_test", "--- Leader Distribution Statistics ---")
	for nodeID, count := range leaderDistribution {
		percentage := float64(count) / float64(rounds) * 100
		utils.LogToFile("vrf_test", fmt.Sprintf("Node %s: selected %d times (%.2f%%)",
			nodeID, count, percentage))
	}

	// Calculate fairness metric (coefficient of variation)
	expectedLeaderCount := float64(rounds) / float64(nodeCount)
	sumSquaredDifference := 0.0
	for _, count := range leaderDistribution {
		diff := float64(count) - expectedLeaderCount
		sumSquaredDifference += diff * diff
	}
	fairnessMetric := 0.0
	if expectedLeaderCount > 0 {
		fairnessMetric = (sumSquaredDifference / float64(nodeCount)) / (expectedLeaderCount * expectedLeaderCount)
	}

	// Log overall statistics
	finalStats := map[string]interface{}{
		"total_rounds":                rounds,
		"node_count":                  nodeCount,
		"successful_verifications":    successfulVerifications,
		"failed_verifications":        failedVerifications,
		"avg_proof_generation_time":   utils.FormatDuration(int64(avgProofGenerationTime)),
		"avg_proof_verification_time": utils.FormatDuration(int64(avgProofVerificationTime)),
		"total_test_time":             utils.FormatDuration(totalTime),
		"fairness_metric":             fmt.Sprintf("%.4f", fairnessMetric),
	}
	utils.LogWithMetadata("vrf_test", "VRF implementation test completed", finalStats)

	return nil
}

// GenerateVRFProof generates a proof for a VRF output
// In a real implementation, this would be a proper VRF proof
func (le *VRFLeaderElection) GenerateVRFProof(round uint64, nodeID string) ([]byte, error) {
	value, err := le.GenerateVRFValue(round, nodeID)
	if err != nil {
		return nil, err
	}

	// In a real VRF, this would be an actual cryptographic proof
	// Here we'll just simulate it with another hash
	h := hmac.New(sha256.New, value)
	h.Write([]byte(strconv.FormatUint(round, 10)))
	proof := h.Sum(nil)

	return proof, nil
}
