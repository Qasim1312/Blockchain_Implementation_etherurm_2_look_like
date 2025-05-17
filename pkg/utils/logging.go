// pkg/utils/logging.go
package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

var (
	// logMutex ensures thread-safe logging
	logMutex sync.Mutex

	// logDirPath is the directory where log files are stored
	logDirPath = "logs"

	// logEnabled determines if logging is enabled
	logEnabled = true

	// initialized is a flag to check if the log directory has been initialized
	initialized = false
)

// Log Categories
const (
	// AMF logging categories
	AMF_SHARDING    = "amf_sharding"
	AMF_REBALANCING = "amf_rebalance"
	AMF_CROSSSHARD  = "amf_crossshard"

	// CAP theorem logging
	CAP_CONSISTENCY  = "cap_consistency"
	CAP_PARTITION    = "cap_partition"
	CAP_AVAILABILITY = "cap_availability"

	// BFT logging
	BFT_REPUTATION = "bft_reputation"
	BFT_DEFENSE    = "bft_defense"
	BFT_CONSENSUS  = "bft_consensus"

	// Verification logging
	VERIFY_PROOF = "verify_proof"
	VERIFY_BLOOM = "verify_bloom"
	VERIFY_ZKP   = "verify_zkp"

	// Synchronization logging
	SYNC_ATOMIC      = "sync_atomic"
	SYNC_HOMOMORPHIC = "sync_homomorphic"
)

// SetLogDirectory sets the directory for log files
func SetLogDirectory(dirPath string) {
	logDirPath = dirPath

	// Create directory if it doesn't exist
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		err := os.MkdirAll(dirPath, 0755)
		if err != nil {
			fmt.Printf("Failed to create log directory: %v\n", err)
		}
	}

	// Create specialized log subdirectories
	createLogCategory(AMF_SHARDING)
	createLogCategory(AMF_REBALANCING)
	createLogCategory(AMF_CROSSSHARD)
	createLogCategory(CAP_CONSISTENCY)
	createLogCategory(CAP_PARTITION)
	createLogCategory(CAP_AVAILABILITY)
	createLogCategory(BFT_REPUTATION)
	createLogCategory(BFT_DEFENSE)
	createLogCategory(BFT_CONSENSUS)
	createLogCategory(VERIFY_PROOF)
	createLogCategory(VERIFY_BLOOM)
	createLogCategory(VERIFY_ZKP)
	createLogCategory(SYNC_ATOMIC)
	createLogCategory(SYNC_HOMOMORPHIC)

	initialized = true
}

// createLogCategory ensures the log category directory exists
func createLogCategory(category string) {
	categoryPath := filepath.Join(logDirPath, category)
	if _, err := os.Stat(categoryPath); os.IsNotExist(err) {
		err := os.MkdirAll(categoryPath, 0755)
		if err != nil {
			fmt.Printf("Failed to create log category directory %s: %v\n", category, err)
		}
	}
}

// EnableLogging enables or disables logging
func EnableLogging(enabled bool) {
	logMutex.Lock()
	defer logMutex.Unlock()
	logEnabled = enabled
}

// IsLoggingEnabled returns whether logging is enabled
func IsLoggingEnabled() bool {
	logMutex.Lock()
	defer logMutex.Unlock()
	return logEnabled
}

// LogToFile logs a message to a file
func LogToFile(fileName, message string) {
	if !logEnabled || !initialized {
		return
	}

	filePath := filepath.Join(logDirPath, fileName+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timestamp := time.Now().Format("2006-01-02 15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timestamp, message)
	f.WriteString(logEntry)
}

// LogWithMetadata logs a message with additional metadata
func LogWithMetadata(logName, message string, metadata map[string]interface{}) error {
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	formattedMessage := fmt.Sprintf("%s | metadata=%s", message, string(metadataJSON))

	LogToFile(logName, formattedMessage)
	return nil
}

// LogPerformanceMetric logs a performance-related metric
func LogPerformanceMetric(component string, operation string, durationNs int64, metadata map[string]interface{}) error {
	if metadata == nil {
		metadata = make(map[string]interface{})
	}

	metadata["duration_ns"] = durationNs
	metadata["duration_ms"] = float64(durationNs) / 1000000.0

	message := fmt.Sprintf("Performance: %s operation completed", operation)
	return LogWithMetadata(component+"_performance", message, metadata)
}

// LogError logs an error with context information
func LogError(component string, operation string, err error, metadata map[string]interface{}) error {
	if metadata == nil {
		metadata = make(map[string]interface{})
	}

	metadata["error"] = err.Error()

	message := fmt.Sprintf("Error in %s operation", operation)
	return LogWithMetadata(component+"_errors", message, metadata)
}

// LogSecurity logs security-related events
func LogSecurity(component string, event string, metadata map[string]interface{}) error {
	message := fmt.Sprintf("Security event: %s", event)
	return LogWithMetadata(component+"_security", message, metadata)
}

// LogDebug logs debug information
func LogDebug(component string, message string, metadata map[string]interface{}) error {
	return LogWithMetadata(component+"_debug", message, metadata)
}

// GetTimestampNano returns the current timestamp in nanoseconds
func GetTimestampNano() int64 {
	return time.Now().UnixNano()
}

// FormatDuration formats a duration in nanoseconds to a human-readable string
func FormatDuration(durationNs int64) string {
	if durationNs < 1000 {
		return fmt.Sprintf("%d ns", durationNs)
	} else if durationNs < 1000000 {
		return fmt.Sprintf("%.2f μs", float64(durationNs)/1000)
	} else if durationNs < 1000000000 {
		return fmt.Sprintf("%.2f ms", float64(durationNs)/1000000)
	} else {
		return fmt.Sprintf("%.2f s", float64(durationNs)/1000000000)
	}
}

// LogAMFSharding logs Adaptive Merkle Forest sharding operations
func LogAMFSharding(message string) {
	if !logEnabled || !initialized {
		return
	}

	// Log to specific AMF sharding log
	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, AMF_SHARDING, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main AMF log for comprehensive view
	LogToFile("amf", fmt.Sprintf("[SHARDING] %s", message))
}

// LogAMFRebalancing logs Adaptive Merkle Forest rebalancing operations
func LogAMFRebalancing(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, AMF_REBALANCING, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main AMF log
	LogToFile("amf", fmt.Sprintf("[REBALANCE] %s", message))
}

// LogAMFCrossShard logs cross-shard operations
func LogAMFCrossShard(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, AMF_CROSSSHARD, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main AMF log
	LogToFile("amf", fmt.Sprintf("[CROSS-SHARD] %s", message))
}

// LogCAPConsistency logs consistency level changes
func LogCAPConsistency(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, CAP_CONSISTENCY, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main CAP log
	LogToFile("cap", fmt.Sprintf("[CONSISTENCY] %s", message))
}

// LogCAPPartition logs network partition probability estimates
func LogCAPPartition(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, CAP_PARTITION, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main CAP log
	LogToFile("cap", fmt.Sprintf("[PARTITION] %s", message))
}

// LogCAPAvailability logs availability metrics
func LogCAPAvailability(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, CAP_AVAILABILITY, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main CAP log
	LogToFile("cap", fmt.Sprintf("[AVAILABILITY] %s", message))
}

// LogBFTReputation logs node reputation score changes
func LogBFTReputation(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, BFT_REPUTATION, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main BFT log
	LogToFile("bft", fmt.Sprintf("[REPUTATION] %s", message))
}

// LogBFTDefense logs defense activations against suspicious behavior
func LogBFTDefense(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, BFT_DEFENSE, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main BFT log
	LogToFile("bft", fmt.Sprintf("[DEFENSE] %s", message))
}

// LogBFTConsensus logs consensus-related information
func LogBFTConsensus(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, BFT_CONSENSUS, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main BFT log
	LogToFile("bft", fmt.Sprintf("[CONSENSUS] %s", message))
}

// LogVerifyProof logs proof compression statistics
func LogVerifyProof(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, VERIFY_PROOF, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main verification log
	LogToFile("verification", fmt.Sprintf("[PROOF] %s", message))
}

// LogVerifyBloom logs Bloom filter operations
func LogVerifyBloom(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, VERIFY_BLOOM, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main verification log
	LogToFile("verification", fmt.Sprintf("[BLOOM] %s", message))
}

// LogVerifyZKP logs zero-knowledge proof operations
func LogVerifyZKP(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, VERIFY_ZKP, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main verification log
	LogToFile("verification", fmt.Sprintf("[ZKP] %s", message))
}

// LogSyncAtomic logs atomic operation completions
func LogSyncAtomic(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, SYNC_ATOMIC, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main sync log
	LogToFile("sync", fmt.Sprintf("[ATOMIC] %s", message))
}

// LogSyncHomomorphic logs homomorphic data structure operations
func LogSyncHomomorphic(message string) {
	if !logEnabled || !initialized {
		return
	}

	timestamp := time.Now().Format("2006-01-02")
	filePath := filepath.Join(logDirPath, SYNC_HOMOMORPHIC, timestamp+".txt")
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	timeStr := time.Now().Format("15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s\n", timeStr, message)
	f.WriteString(logEntry)

	// Also log to main sync log
	LogToFile("sync", fmt.Sprintf("[HOMOMORPHIC] %s", message))
}
