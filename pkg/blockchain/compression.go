// pkg/blockchain/compression.go
package blockchain

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"encoding/gob"
	"io/ioutil"
	"time"
)

// CompressedStateBlob is a compact representation of the state
type CompressedStateBlob struct {
	RootHash string
	Data     []byte
}

// CompressedState represents a compressed blockchain state
type CompressedState struct {
	RootHash       string
	PrunedAccounts map[string]bool // Accounts that were pruned
	DeltaEncoding  []byte          // Delta from previous state
	Timestamp      time.Time
	Version        uint32
}

// PruningStrategy defines how state is pruned
type PruningStrategy struct {
	KeepEveryNth        int           // Keep every Nth state fully
	MinimumAge          time.Duration // Minimum age before pruning
	ImportanceThreshold float64       // Importance threshold for pruning
}

// CompressState compresses state data for storage efficiency
func CompressState(state *State) ([]byte, error) {
	// Create a buffer to hold the compressed data
	var buf bytes.Buffer

	// Create a zlib writer for compression
	zw := zlib.NewWriter(&buf)

	// Use gob encoding for serialization
	enc := gob.NewEncoder(zw)

	// Get a snapshot of accounts
	snapshot := state.CreateSnapshot()

	// Encode account data
	if err := enc.Encode(snapshot); err != nil {
		return nil, err
	}

	// Make sure to flush and close the zlib writer
	zw.Close()

	return buf.Bytes(), nil
}

// DecompressState decompresses state data
func DecompressState(data []byte) (map[string]*Account, error) {
	// Create a reader for the compressed data
	zr, err := zlib.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer zr.Close()

	// Read the compressed data
	uncompressed, err := ioutil.ReadAll(zr)
	if err != nil {
		return nil, err
	}

	// Decode the decompressed data
	dec := gob.NewDecoder(bytes.NewReader(uncompressed))
	var accounts map[string]*Account
	if err := dec.Decode(&accounts); err != nil {
		return nil, err
	}

	return accounts, nil
}

// CalculateDelta calculates the delta between two states
func CalculateDelta(oldState, newState *State) []byte {
	// Get snapshots of both states
	oldSnapshot := oldState.CreateSnapshot()
	newSnapshot := newState.CreateSnapshot()

	// Create a buffer to hold the delta
	var buf bytes.Buffer

	// Record account additions, modifications, and deletions
	for addr, newAcc := range newSnapshot {
		oldAcc, exists := oldSnapshot[addr]
		if !exists {
			// Account was added
			binary.Write(&buf, binary.LittleEndian, byte(1)) // 1 = added
			buf.WriteString(addr)

			// Write account data
			accBytes, _ := encodeAccount(newAcc)
			binary.Write(&buf, binary.LittleEndian, uint32(len(accBytes)))
			buf.Write(accBytes)
		} else if !accountsEqual(oldAcc, newAcc) {
			// Account was modified
			binary.Write(&buf, binary.LittleEndian, byte(2)) // 2 = modified
			buf.WriteString(addr)

			// Write account data
			accBytes, _ := encodeAccount(newAcc)
			binary.Write(&buf, binary.LittleEndian, uint32(len(accBytes)))
			buf.Write(accBytes)
		}
	}

	// Record deletions
	for addr := range oldSnapshot {
		if _, exists := newSnapshot[addr]; !exists {
			// Account was deleted
			binary.Write(&buf, binary.LittleEndian, byte(3)) // 3 = deleted
			buf.WriteString(addr)
		}
	}

	return buf.Bytes()
}

// encodeAccount serializes an account to bytes
func encodeAccount(acc *Account) ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(acc); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// accountsEqual checks if two accounts are identical
func accountsEqual(a, b *Account) bool {
	if a.Address != b.Address || a.Balance != b.Balance || a.Nonce != b.Nonce {
		return false
	}

	// Compare storage maps
	if len(a.Storage) != len(b.Storage) {
		return false
	}

	for k, v := range a.Storage {
		if bv, exists := b.Storage[k]; !exists || bv != v {
			return false
		}
	}

	return true
}
