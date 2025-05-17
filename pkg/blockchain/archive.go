package blockchain

import (
	"bytes"
	"compress/zlib"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// StateArchiver handles state compression, archival, and pruning
type StateArchiver struct {
	archivePath     string
	storageLimit    int64 // Max storage in bytes
	retentionPeriod time.Duration
	compressionRate int // 0-9, higher means more compression
	mutex           sync.RWMutex
	stateHashes     map[string]string // Maps archive references to state root hashes
	metadata        *ArchiveMetadata
}

// ArchiveMetadata contains information about archives
type ArchiveMetadata struct {
	Archives         map[string]*ArchiveInfo // Map of archive ID to info
	TotalSize        int64
	LastPruneTime    time.Time
	VerificationKeys map[string][]byte // Integrity verification keys
	mutex            sync.RWMutex
}

// ArchiveInfo contains metadata about a specific archive
type ArchiveInfo struct {
	ID           string
	CreatedAt    time.Time
	StateRoot    string
	Size         int64
	BlockHeight  uint64
	Filename     string
	Checksum     string
	IntegrityTag []byte // HMAC or similar for integrity verification
}

// NewStateArchiver creates a new state archiver
func NewStateArchiver(archivePath string) *StateArchiver {
	// Create archive directory if it doesn't exist
	if err := os.MkdirAll(archivePath, 0755); err != nil {
		log.Printf("[Archive] Error creating archive directory: %v", err)
	}

	return &StateArchiver{
		archivePath:     archivePath,
		storageLimit:    1024 * 1024 * 1024 * 10, // 10 GB default limit
		retentionPeriod: 30 * 24 * time.Hour,     // 30 days default retention
		compressionRate: 6,                       // Medium compression by default
		stateHashes:     make(map[string]string),
		metadata: &ArchiveMetadata{
			Archives:         make(map[string]*ArchiveInfo),
			VerificationKeys: make(map[string][]byte),
			LastPruneTime:    time.Now(),
		},
	}
}

// ArchiveState compresses and archives the current state
func (sa *StateArchiver) ArchiveState(state *State, blockHeight uint64) (string, error) {
	sa.mutex.Lock()
	defer sa.mutex.Unlock()

	// Generate archive ID
	timestamp := time.Now().UnixNano()
	archiveID := fmt.Sprintf("archive-%d-%s", timestamp, state.GetRootHash()[:8])
	filename := filepath.Join(sa.archivePath, archiveID+".state")

	// Create compressed representation of state
	compressedData, err := sa.compressState(state)
	if err != nil {
		return "", fmt.Errorf("failed to compress state: %v", err)
	}

	// Generate integrity tag
	integrityTag := sa.generateIntegrityTag(compressedData, []byte(state.GetRootHash()))

	// Write to file
	if err := os.WriteFile(filename, compressedData, 0644); err != nil {
		return "", fmt.Errorf("failed to write archive file: %v", err)
	}

	// Calculate checksum
	checksum := sha256.Sum256(compressedData)
	checksumHex := hex.EncodeToString(checksum[:])

	// Store metadata
	info := &ArchiveInfo{
		ID:           archiveID,
		CreatedAt:    time.Now(),
		StateRoot:    state.GetRootHash(),
		Size:         int64(len(compressedData)),
		BlockHeight:  blockHeight,
		Filename:     filename,
		Checksum:     checksumHex,
		IntegrityTag: integrityTag,
	}

	sa.metadata.mutex.Lock()
	sa.metadata.Archives[archiveID] = info
	sa.metadata.TotalSize += info.Size
	sa.metadata.mutex.Unlock()

	// Map state root hash to archive ID
	sa.stateHashes[state.GetRootHash()] = archiveID

	log.Printf("[Archive] State archived: %s (%.2f MB), root hash: %s",
		archiveID, float64(info.Size)/(1024*1024), state.GetRootHash()[:8])

	// Check if pruning is needed
	if sa.metadata.TotalSize > sa.storageLimit {
		go sa.pruneArchives()
	}

	return archiveID, nil
}

// compressState compresses the state data
func (sa *StateArchiver) compressState(state *State) ([]byte, error) {
	// Serialize state
	serialized, err := state.Serialize()
	if err != nil {
		return nil, err
	}

	// Compress data
	var compressed bytes.Buffer
	writer, err := zlib.NewWriterLevel(&compressed, sa.compressionRate)
	if err != nil {
		return nil, err
	}

	if _, err := writer.Write(serialized); err != nil {
		return nil, err
	}
	writer.Close()

	return compressed.Bytes(), nil
}

// RestoreState restores state from an archive
func (sa *StateArchiver) RestoreState(archiveID string) (*State, error) {
	sa.mutex.RLock()
	defer sa.mutex.RUnlock()

	// Find archive info
	sa.metadata.mutex.RLock()
	info, exists := sa.metadata.Archives[archiveID]
	sa.metadata.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("archive not found: %s", archiveID)
	}

	// Read compressed data
	compressedData, err := os.ReadFile(info.Filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read archive file: %v", err)
	}

	// Verify checksum
	checksum := sha256.Sum256(compressedData)
	checksumHex := hex.EncodeToString(checksum[:])
	if checksumHex != info.Checksum {
		return nil, errors.New("archive integrity check failed: checksum mismatch")
	}

	// Verify integrity tag
	if !sa.verifyIntegrityTag(compressedData, []byte(info.StateRoot), info.IntegrityTag) {
		return nil, errors.New("archive integrity check failed: integrity tag mismatch")
	}

	// Decompress data
	reader, err := zlib.NewReader(bytes.NewReader(compressedData))
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var serialized bytes.Buffer
	if _, err := io.Copy(&serialized, reader); err != nil {
		return nil, err
	}

	// Deserialize state
	state := &State{}
	if err := state.Deserialize(serialized.Bytes()); err != nil {
		return nil, err
	}

	// Verify state root hash
	if state.GetRootHash() != info.StateRoot {
		return nil, errors.New("state root hash mismatch after restore")
	}

	log.Printf("[Archive] State restored from archive: %s, root hash: %s",
		archiveID, info.StateRoot[:8])

	return state, nil
}

// pruneArchives removes old archives based on retention policy
func (sa *StateArchiver) pruneArchives() {
	sa.mutex.Lock()
	defer sa.mutex.Unlock()

	sa.metadata.mutex.Lock()
	defer sa.metadata.mutex.Unlock()

	log.Printf("[Archive] Starting pruning operation. Current size: %.2f GB, limit: %.2f GB",
		float64(sa.metadata.TotalSize)/(1024*1024*1024), float64(sa.storageLimit)/(1024*1024*1024))

	// Find candidates for pruning
	var candidates []*ArchiveInfo
	for _, info := range sa.metadata.Archives {
		// Skip if within retention period
		if time.Since(info.CreatedAt) < sa.retentionPeriod {
			continue
		}
		candidates = append(candidates, info)
	}

	// Sort candidates by age (oldest first)
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[i].CreatedAt.After(candidates[j].CreatedAt) {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Calculate how much we need to prune
	var targetSize int64
	if sa.metadata.TotalSize > sa.storageLimit {
		// Aim to get to 80% of limit to avoid frequent pruning
		targetSize = sa.metadata.TotalSize - int64(float64(sa.storageLimit)*0.8)
	} else {
		// No need to prune if under limit
		return
	}

	var prunedSize int64
	var prunedCount int

	// Prune archives until we reach target size
	for _, info := range candidates {
		if prunedSize >= targetSize {
			break
		}

		// Delete file
		if err := os.Remove(info.Filename); err != nil {
			log.Printf("[Archive] Error deleting archive file %s: %v", info.Filename, err)
			continue
		}

		// Update metadata
		prunedSize += info.Size
		prunedCount++
		sa.metadata.TotalSize -= info.Size
		delete(sa.metadata.Archives, info.ID)
		delete(sa.stateHashes, info.StateRoot)

		log.Printf("[Archive] Pruned archive: %s (%.2f MB)", info.ID, float64(info.Size)/(1024*1024))
	}

	sa.metadata.LastPruneTime = time.Now()

	log.Printf("[Archive] Pruning complete. Removed %d archives (%.2f GB). New size: %.2f GB",
		prunedCount, float64(prunedSize)/(1024*1024*1024), float64(sa.metadata.TotalSize)/(1024*1024*1024))
}

// generateIntegrityTag creates an integrity tag for the compressed data
func (sa *StateArchiver) generateIntegrityTag(data []byte, key []byte) []byte {
	// Generate HMAC-like integrity tag
	h := sha256.New()
	h.Write(key)
	h.Write(data)
	h.Write(key) // Sandwich pattern for added security
	return h.Sum(nil)
}

// verifyIntegrityTag verifies the integrity tag
func (sa *StateArchiver) verifyIntegrityTag(data []byte, key []byte, tag []byte) bool {
	// Regenerate tag and compare
	h := sha256.New()
	h.Write(key)
	h.Write(data)
	h.Write(key)
	calculatedTag := h.Sum(nil)
	return bytes.Equal(calculatedTag, tag)
}

// GetArchiveInfo returns information about a specific archive
func (sa *StateArchiver) GetArchiveInfo(archiveID string) (*ArchiveInfo, error) {
	sa.metadata.mutex.RLock()
	defer sa.metadata.mutex.RUnlock()

	info, exists := sa.metadata.Archives[archiveID]
	if !exists {
		return nil, fmt.Errorf("archive not found: %s", archiveID)
	}

	return info, nil
}

// GetArchiveIDByStateRoot returns the archive ID for a state root hash
func (sa *StateArchiver) GetArchiveIDByStateRoot(stateRoot string) (string, error) {
	sa.mutex.RLock()
	defer sa.mutex.RUnlock()

	archiveID, exists := sa.stateHashes[stateRoot]
	if !exists {
		return "", fmt.Errorf("no archive found for state root: %s", stateRoot)
	}

	return archiveID, nil
}

// GetArchiveStats returns statistics about archived states
func (sa *StateArchiver) GetArchiveStats() map[string]interface{} {
	sa.mutex.RLock()
	defer sa.mutex.RUnlock()
	sa.metadata.mutex.RLock()
	defer sa.metadata.mutex.RUnlock()

	return map[string]interface{}{
		"totalArchives":     len(sa.metadata.Archives),
		"totalSizeBytes":    sa.metadata.TotalSize,
		"totalSizeGB":       float64(sa.metadata.TotalSize) / (1024 * 1024 * 1024),
		"storageLimitGB":    float64(sa.storageLimit) / (1024 * 1024 * 1024),
		"lastPruneTime":     sa.metadata.LastPruneTime,
		"compressionLevel":  sa.compressionRate,
		"retentionPeriodHr": sa.retentionPeriod.Hours(),
	}
}

// GetArchivedStateRoots returns all available state root hashes
func (sa *StateArchiver) GetArchivedStateRoots() []string {
	sa.mutex.RLock()
	defer sa.mutex.RUnlock()

	roots := make([]string, 0, len(sa.stateHashes))
	for root := range sa.stateHashes {
		roots = append(roots, root)
	}
	return roots
}

// StateSnapshot represents a lightweight state snapshot for quick access
type StateSnapshot struct {
	RootHash       string
	BlockHeight    uint64
	AccountCount   int
	TotalBalance   uint64
	SnapshotTime   time.Time
	ArchiveID      string
	IntegrityProof []byte
}

// CreateStateSnapshot creates a compact snapshot of the current state
func (sa *StateArchiver) CreateStateSnapshot(state *State, blockHeight uint64) (*StateSnapshot, error) {
	// First archive the state
	archiveID, err := sa.ArchiveState(state, blockHeight)
	if err != nil {
		return nil, err
	}

	// Get total balance (simplified implementation)
	var totalBalance uint64
	accounts := state.GetAccounts()
	for _, acc := range accounts {
		totalBalance += acc.Balance
	}

	// Create snapshot
	snapshot := &StateSnapshot{
		RootHash:     state.GetRootHash(),
		BlockHeight:  blockHeight,
		AccountCount: len(accounts),
		TotalBalance: totalBalance,
		SnapshotTime: time.Now(),
		ArchiveID:    archiveID,
	}

	// Generate integrity proof
	h := sha256.New()
	h.Write([]byte(snapshot.RootHash))
	binary.Write(h, binary.BigEndian, snapshot.BlockHeight)
	binary.Write(h, binary.BigEndian, snapshot.AccountCount)
	binary.Write(h, binary.BigEndian, snapshot.TotalBalance)
	binary.Write(h, binary.BigEndian, snapshot.SnapshotTime.Unix())
	h.Write([]byte(snapshot.ArchiveID))
	snapshot.IntegrityProof = h.Sum(nil)

	return snapshot, nil
}

// VerifyStateSnapshot verifies the integrity of a state snapshot
func (sa *StateArchiver) VerifyStateSnapshot(snapshot *StateSnapshot) bool {
	// Recalculate integrity proof
	h := sha256.New()
	h.Write([]byte(snapshot.RootHash))
	binary.Write(h, binary.BigEndian, snapshot.BlockHeight)
	binary.Write(h, binary.BigEndian, snapshot.AccountCount)
	binary.Write(h, binary.BigEndian, snapshot.TotalBalance)
	binary.Write(h, binary.BigEndian, snapshot.SnapshotTime.Unix())
	h.Write([]byte(snapshot.ArchiveID))
	calculatedProof := h.Sum(nil)

	return bytes.Equal(calculatedProof, snapshot.IntegrityProof)
}

// RestoreFromSnapshot restores a state from a snapshot
func (sa *StateArchiver) RestoreFromSnapshot(snapshot *StateSnapshot) (*State, error) {
	// Verify snapshot integrity
	if !sa.VerifyStateSnapshot(snapshot) {
		return nil, errors.New("snapshot integrity verification failed")
	}

	// Use the archive ID from the snapshot to restore the state
	return sa.RestoreState(snapshot.ArchiveID)
}

// SetRetentionPolicy sets the retention policy for archived states
func (sa *StateArchiver) SetRetentionPolicy(retentionPeriod time.Duration, storageLimit int64) {
	sa.mutex.Lock()
	defer sa.mutex.Unlock()

	sa.retentionPeriod = retentionPeriod
	sa.storageLimit = storageLimit

	log.Printf("[Archive] Retention policy updated: period=%v, limit=%.2f GB",
		retentionPeriod, float64(storageLimit)/(1024*1024*1024))
}

// SetCompressionLevel sets the compression level for state archival
func (sa *StateArchiver) SetCompressionLevel(level int) error {
	if level < 0 || level > 9 {
		return errors.New("compression level must be between 0 and 9")
	}

	sa.mutex.Lock()
	defer sa.mutex.Unlock()

	sa.compressionRate = level
	return nil
}
