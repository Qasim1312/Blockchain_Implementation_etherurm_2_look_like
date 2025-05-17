// pkg/blockchain/state_archival.go
package blockchain

import (
	"fmt"
	"log"
	"time"
)

// StateBlob represents a compressed state for archival
type StateBlob struct {
	RootHash       string
	CompressedData []byte
	Timestamp      int64
}

// StateArchiverInterface handles state compression and archival
type StateArchiverInterface interface {
	ArchiveCurrentState()
	PruneOldStates()
}

// SimpleStateArchiver implements the StateArchiverInterface
type SimpleStateArchiver struct {
	state       *State
	archives    []*StateBlob
	lastArchive time.Time
}

// NewSimpleStateArchiver creates a new state archiver
func NewSimpleStateArchiver(state *State) StateArchiverInterface {
	return &SimpleStateArchiver{
		state:       state,
		archives:    make([]*StateBlob, 0),
		lastArchive: time.Now(),
	}
}

// ArchiveCurrentState archives the current state
func (sa *SimpleStateArchiver) ArchiveCurrentState() {
	// Only archive if significant time has passed
	if time.Since(sa.lastArchive) < 10*time.Minute {
		return
	}

	stateBlob, err := SerializeState(sa.state)
	if err != nil {
		log.Printf("Failed to archive state: %v", err)
		return
	}

	sa.archives = append(sa.archives, stateBlob)
	sa.lastArchive = time.Now()

	log.Printf("State archived at time %v with root %s, compressed size %d bytes",
		time.Unix(stateBlob.Timestamp, 0), stateBlob.RootHash, len(stateBlob.CompressedData))
}

// PruneOldStates removes old state archives
func (sa *SimpleStateArchiver) PruneOldStates() {
	if len(sa.archives) <= 10 {
		return // Keep at least 10 archives
	}

	// Remove all but the 10 most recent archives
	sa.archives = sa.archives[len(sa.archives)-10:]

	log.Printf("State archives pruned, keeping %d most recent archives", len(sa.archives))
}

// SerializeState compresses a state for storage
func SerializeState(state *State) (*StateBlob, error) {
	// In a real implementation, this would compress the state data
	// For this simulation, we'll just create a blob with the root hash
	blob := &StateBlob{
		RootHash:       state.GetRootHash(),
		CompressedData: []byte(fmt.Sprintf("Compressed state data for root %s", state.GetRootHash())),
		Timestamp:      time.Now().Unix(),
	}

	return blob, nil
}
