// pkg/blockchain/chain.go
package blockchain

import (
	"time"
)

// Blockchain represents the main blockchain structure
type Blockchain struct {
	Blocks         []*Block
	AdvancedBlocks []*AdvancedBlock
	CurrentState   *State
	StateArchiver  StateArchiverInterface
	ZKVerifier     *ZKVerifier
}

// NewBlockchain creates a new blockchain with genesis block
func NewBlockchain() *Blockchain {
	// Create initial state
	state := NewState()

	// Create genesis block
	genesis := NewBlock(0, []byte("Genesis Block"), "", []Transaction{})

	// Create advanced genesis block with additional features
	advGenesis := NewAdvancedBlock(0, []byte("Genesis Block"), "", []Transaction{}, state)

	// Create state archiver
	archiver := NewSimpleStateArchiver(state)

	// Create ZK verifier
	zkVerifier := NewZKVerifier()

	return &Blockchain{
		Blocks:         []*Block{genesis},
		AdvancedBlocks: []*AdvancedBlock{advGenesis},
		CurrentState:   state,
		StateArchiver:  archiver,
		ZKVerifier:     zkVerifier,
	}
}

// AddBlock adds a new block to the blockchain with advanced features
func (bc *Blockchain) AddBlock(data []byte, transactions []Transaction) *AdvancedBlock {
	// Get the previous block
	prevBlock := bc.Blocks[len(bc.Blocks)-1]

	// Get next height
	nextHeight := uint64(len(bc.Blocks))

	// Create basic block
	newBlock := NewBlock(nextHeight, data, prevBlock.Hash, transactions)

	// Update state with new transactions
	for _, tx := range transactions {
		bc.CurrentState.ApplyTransaction(&tx)
	}
	newBlock.StateRoot = bc.CurrentState.GetRootHash()

	// Create advanced block with additional features
	advancedBlock := NewAdvancedBlock(nextHeight, data, prevBlock.Hash, transactions, bc.CurrentState)

	// Archive current state
	bc.StateArchiver.ArchiveCurrentState()

	// Add blocks to chain
	bc.Blocks = append(bc.Blocks, newBlock)
	bc.AdvancedBlocks = append(bc.AdvancedBlocks, advancedBlock)

	return advancedBlock
}

// AddBlockWithNonce adds a new block to the blockchain with advanced features and specified nonce
func (bc *Blockchain) AddBlockWithNonce(data []byte, transactions []Transaction, nonce uint64) *AdvancedBlock {
	// Get the previous block
	prevBlock := bc.Blocks[len(bc.Blocks)-1]

	// Get next height
	nextHeight := uint64(len(bc.Blocks))

	// Create basic block with the specified nonce
	newBlock := NewBlock(nextHeight, data, prevBlock.Hash, transactions)
	newBlock.Nonce = nonce
	newBlock.Hash = newBlock.CalculateHash() // Recalculate hash with the provided nonce

	// Update state with new transactions
	for _, tx := range transactions {
		bc.CurrentState.ApplyTransaction(&tx)
	}
	newBlock.StateRoot = bc.CurrentState.GetRootHash()

	// Create advanced block with additional features and pass the nonce
	advancedBlock := NewAdvancedBlockWithNonce(nextHeight, data, prevBlock.Hash, transactions, bc.CurrentState, nonce)

	// Archive current state
	bc.StateArchiver.ArchiveCurrentState()

	// Add blocks to chain
	bc.Blocks = append(bc.Blocks, newBlock)
	bc.AdvancedBlocks = append(bc.AdvancedBlocks, advancedBlock)

	return advancedBlock
}

// GetLatestBlock returns the latest block in the chain
func (bc *Blockchain) GetLatestBlock() *Block {
	return bc.Blocks[len(bc.Blocks)-1]
}

// GetLatestAdvancedBlock returns the latest advanced block in the chain
func (bc *Blockchain) GetLatestAdvancedBlock() *AdvancedBlock {
	return bc.AdvancedBlocks[len(bc.AdvancedBlocks)-1]
}

// VerifyBlockEntropy checks if a block's entropy meets security requirements
func (bc *Blockchain) VerifyBlockEntropy(block *AdvancedBlock) bool {
	// Blocks with entropy score below 0.5 are considered insecure
	passed := block.EntropyScore >= 0.5

	if passed {
		println("[INFO] Block passed entropy verification with score:", block.EntropyScore)
	} else {
		println("[WARN] Block failed entropy verification with score:", block.EntropyScore)
	}

	return passed
}

// GenerateStateProof generates a zero-knowledge proof for an account
func (bc *Blockchain) GenerateStateProof(account string) (*ZKStateProof, error) {
	return bc.ZKVerifier.GenerateStateProof(bc.CurrentState, account)
}

// VerifyStateProof verifies a state proof
func (bc *Blockchain) VerifyStateProof(proof *ZKStateProof) bool {
	return bc.ZKVerifier.VerifyStateProof(proof)
}

// CompressAndArchiveState compresses the current state for long-term storage
func (bc *Blockchain) CompressAndArchiveState() (*StateBlob, error) {
	return SerializeState(bc.CurrentState)
}

// PruneOldStates removes unnecessary historical states
func (bc *Blockchain) PruneOldStates() {
	// Only prune if we have enough history
	if len(bc.Blocks) > 100 {
		bc.StateArchiver.PruneOldStates()
	}
}

// PerformPeriodicMaintenance performs regular maintenance tasks
func (bc *Blockchain) PerformPeriodicMaintenance() {
	// Archive state
	bc.StateArchiver.ArchiveCurrentState()

	// Prune old states
	if time.Now().Hour() == 2 { // Do at 2 AM
		bc.PruneOldStates()
	}
}

// DemonstrateZKProofs demonstrates zero-knowledge proof operations for a demo
func (bc *Blockchain) DemonstrateZKProofs() {
	// Create a test account with some balance
	testAddr := "0x1234567890abcdef1234567890abcdef12345678"

	// First ensure account exists
	account, err := bc.CurrentState.GetAccount(testAddr)
	if err != nil {
		// Create the account if it doesn't exist
		account, err = bc.CurrentState.CreateAccount(testAddr)
		if err != nil {
			println("[DEMO] Failed to create test account:", err.Error())
			return
		}
	}

	// Update account with test balance
	account.Balance = 1000
	account.Nonce = 5
	bc.CurrentState.UpdateAccount(account)

	println("[DEMO] Demonstrating Zero-Knowledge Proof operations")

	// Generate a proof for the account
	proof, err := bc.GenerateStateProof(testAddr)
	if err != nil {
		println("[DEMO] Failed to generate proof:", err.Error())
		return
	}

	// Verify the proof
	verified := bc.VerifyStateProof(proof)
	if verified {
		println("[DEMO] ZKP verification successful!")
	} else {
		println("[DEMO] ZKP verification failed!")
	}

	// Try to verify the same proof again (should fail due to replay protection)
	verified = bc.VerifyStateProof(proof)
	if verified {
		println("[DEMO] ZKP verification successful on repeat (unexpected)")
	} else {
		println("[DEMO] ZKP verification correctly failed on repeat due to replay protection")
	}
}

// ReplaceChain replaces the current blockchain with a new one if it's longer and valid
func (bc *Blockchain) ReplaceChain(newBlocks []*Block, newAdvBlocks []*AdvancedBlock) bool {
	// Sanity check: Both slice lengths should match
	if len(newBlocks) != len(newAdvBlocks) {
		return false
	}

	// Verify chain integrity - each block should point to the previous block
	for i := 1; i < len(newBlocks); i++ {
		if newBlocks[i].PrevHash != newBlocks[i-1].Hash {
			return false // Chain is invalid - blocks don't connect properly
		}
	}

	// Check if the new chain is longer and valid
	if len(newBlocks) > len(bc.Blocks) {
		// Backup our current state in case we need to revert

		// Sanity check: Make sure genesis blocks match to avoid invalid chain replacements
		if len(bc.Blocks) > 0 && len(newBlocks) > 0 && bc.Blocks[0].Hash != newBlocks[0].Hash {
			return false // Different genesis blocks, can't replace
		}

		// Replace the blockchain
		bc.Blocks = newBlocks
		bc.AdvancedBlocks = newAdvBlocks

		// Reset and rebuild state
		bc.CurrentState = NewState()

		// Replay all transactions to rebuild the state
		for _, block := range bc.Blocks {
			for _, tx := range block.Transactions {
				bc.CurrentState.ApplyTransaction(&tx)
			}
		}

		return true
	}

	return false
}
