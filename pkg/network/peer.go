package network

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"math/rand"
	"time"

	bhost "github.com/libp2p/go-libp2p"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	ma "github.com/multiformats/go-multiaddr"

	"github.com/qasim/blockchain_assignment_3/pkg/blockchain"
	"github.com/qasim/blockchain_assignment_3/pkg/consensus"
)

// PeerDiscoveryTopic is used for peer discovery announcements
const PeerDiscoveryTopic = "peer-discovery"

// Node wraps libp2p + gossipsub + consensus + local chain.
type Node struct {
	Ctx       context.Context
	Host      host.Host
	PubSub    *pubsub.PubSub
	BlockSub  *pubsub.Subscription
	Chain     *blockchain.Blockchain
	Consensus *consensus.HybridConsensus
	IsBootstr bool
}

// NetSync handles network synchronization
type NetSync struct {
	node           *Node
	discoveryPeers map[string]bool
	lastSync       time.Time
}

// New creates and boots a full node.
func New(ctx context.Context, port int, bootstrap []string, id string) (*Node, error) {
	h, err := bhost.New(bhost.ListenAddrStrings(
		fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", port)))
	if err != nil {
		return nil, err
	}

	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		return nil, err
	}
	sub, err := ps.Subscribe(BlocksTopic)
	if err != nil {
		return nil, err
	}

	chain := blockchain.NewBlockchain()
	cons, err := consensus.NewHybridConsensus(id)
	if err != nil {
		return nil, err
	}
	cons.SetAsValidator(true)

	n := &Node{
		Ctx:       ctx,
		Host:      h,
		PubSub:    ps,
		BlockSub:  sub,
		Chain:     chain,
		Consensus: cons,
	}

	// Dial bootstrap peers if supplied.
	for _, addr := range bootstrap {
		a, err := ma.NewMultiaddr(addr)
		if err != nil {
			log.Printf("bad bootstrap addr %s: %v", addr, err)
			continue
		}
		pi, err := peer.AddrInfoFromP2pAddr(a)
		if err == nil {
			_ = h.Connect(ctx, *pi)
		}
	}

	// Launch background reactors.
	go n.blockProducer()
	go n.blockConsumer()

	return n, nil
}

// Wire format.
type wireBlock struct {
	Block *blockchain.Block
}

// blockchainToConsensusBlock converts a blockchain.Block to a consensus.Block
func blockchainToConsensusBlock(block *blockchain.Block) *consensus.Block {
	return &consensus.Block{
		Index:     block.Index,
		Timestamp: block.Timestamp,
		PrevHash:  block.PrevHash,
		Hash:      block.Hash,
		Data:      block.Data,
		Nonce:     block.Nonce,
		Producer:  "", // No Producer field in blockchain.Block
	}
}

// Leader (≈10 % chance) produces and broadcasts a block each round.
func (n *Node) blockProducer() {
	ticker := time.NewTicker(15 * time.Second)
	for range ticker.C {
		n.Consensus.StartNewRound()

		// First, check if we might be the leader
		// This is a simplified approach - in a real implementation we'd wait for proper leader election
		if rand.Float64() < 0.1 { // 10% chance of being leader
			// Create a block directly instead of relying on consensus's proposed block
			prevBlock := n.Chain.GetLatestBlock()

			newBlock := &blockchain.Block{
				Index:        prevBlock.Index + 1,
				Timestamp:    time.Now().Unix(),
				Data:         []byte(fmt.Sprintf("Block data for round %d", prevBlock.Index+1)),
				PrevHash:     prevBlock.Hash,
				Nonce:        0, // Will be set during mining
				StateRoot:    n.Chain.CurrentState.GetRootHash(),
				Transactions: []blockchain.Transaction{}, // Empty transactions
			}

			// Perform a basic PoW (simplified for demonstration)
			for i := uint64(0); i < 10000; i++ {
				newBlock.Nonce = i
				hash := sha256.Sum256([]byte(fmt.Sprintf("%d%d%s%s%d",
					newBlock.Index, newBlock.Timestamp, newBlock.Data, newBlock.PrevHash, newBlock.Nonce)))
				newBlock.Hash = hex.EncodeToString(hash[:])
				break // In a real implementation, we'd check if the hash meets difficulty
			}

			// Broadcast block
			msg, _ := json.Marshal(wireBlock{Block: newBlock})
			_ = n.PubSub.Publish(BlocksTopic, msg)

			log.Printf("[LEADER] Broadcasting new block: height=%d hash=%s",
				newBlock.Index, newBlock.Hash)
		}
	}
}

// NewAdvanced creates and boots a full node with advanced blockchain features
func NewAdvanced(ctx context.Context, port int, bootstrap []string, id string) (*Node, error) {
	h, err := bhost.New(bhost.ListenAddrStrings(
		fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", port)))
	if err != nil {
		return nil, err
	}

	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		return nil, err
	}
	sub, err := ps.Subscribe(BlocksTopic)
	if err != nil {
		return nil, err
	}

	// Create blockchain with advanced features
	chain := blockchain.NewBlockchain()
	cons, err := consensus.NewHybridConsensus(id)
	if err != nil {
		return nil, err
	}
	cons.SetAsValidator(true)

	n := &Node{
		Ctx:       ctx,
		Host:      h,
		PubSub:    ps,
		BlockSub:  sub,
		Chain:     chain,
		Consensus: cons,
	}

	// Wait a short time to allow host to initialize before connecting
	time.Sleep(500 * time.Millisecond)

	// Create and start network sync
	netSync := NewNetSync(n)
	go netSync.Start()

	// Connect to bootstrap nodes first
	for _, addr := range bootstrap {
		if addr == "" {
			continue
		}
		a, err := ma.NewMultiaddr(addr)
		if err != nil {
			log.Printf("[ERROR] Bad bootstrap addr %s: %v", addr, err)
			continue
		}
		pi, err := peer.AddrInfoFromP2pAddr(a)
		if err != nil {
			log.Printf("[ERROR] Failed to parse peer address from %s: %v", addr, err)
			continue
		}

		log.Printf("[CONNECT] Connecting to bootstrap peer: %s", a)
		if err := h.Connect(ctx, *pi); err != nil {
			log.Printf("[ERROR] Failed to connect to bootstrap peer %s: %v", a, err)
		} else {
			log.Printf("[CONNECT] Successfully connected to peer: %s", pi.ID.String()[:12])

			// Wait a bit to establish connection fully
			time.Sleep(100 * time.Millisecond)

			// Immediately broadcast our genesis block to sync chains from the start
			if len(chain.Blocks) > 0 {
				genesisBlock := chain.Blocks[0]
				log.Printf("[SYNC] Broadcasting genesis block: hash=%s", genesisBlock.Hash)
				msg, _ := json.Marshal(wireBlock{Block: genesisBlock})
				if err := n.PubSub.Publish(BlocksTopic, msg); err != nil {
					log.Printf("[ERROR] Failed to publish genesis block: %v", err)
				}
			}
		}
	}

	// Launch background reactors
	go n.advancedBlockProducer() // Use advanced block producer
	go n.advancedBlockConsumer() // Use advanced block consumer

	return n, nil
}

// Wire format for advanced blocks
type wireAdvancedBlock struct {
	Block *blockchain.AdvancedBlock
}

// Advanced block production
func (n *Node) advancedBlockProducer() {
	ticker := time.NewTicker(15 * time.Second)
	for range ticker.C {
		n.Consensus.StartNewRound()

		// First, check if we might be the leader
		// This is a simplified approach - in a real implementation we'd wait for proper leader election
		if rand.Float64() < 0.1 { // 10% chance of being leader
			// Create a block directly instead of relying on consensus's proposed block
			if n.Chain == nil || n.Chain.GetLatestBlock() == nil || n.Chain.CurrentState == nil {
				log.Printf("[ERROR] Chain is not properly initialized")
				continue
			}

			prevBlock := n.Chain.GetLatestBlock()

			// Get current height properly
			nextHeight := prevBlock.Index + 1

			// Create some random transactions for this block
			numTxs := rand.Intn(5) + 1 // 1-5 transactions
			transactions := make([]blockchain.Transaction, numTxs)

			for i := 0; i < numTxs; i++ {
				// Create a transaction with random data
				from := fmt.Sprintf("user%d", rand.Intn(10))
				to := fmt.Sprintf("user%d", rand.Intn(10))
				for to == from {
					to = fmt.Sprintf("user%d", rand.Intn(10)) // Ensure different addresses
				}

				transactions[i] = blockchain.Transaction{
					ID:        fmt.Sprintf("tx-%d-%d", nextHeight, i),
					From:      from,
					To:        to,
					Value:     uint64(rand.Intn(100) + 1), // 1-100 value
					Data:      []byte(fmt.Sprintf("Transaction data %d", i)),
					Timestamp: time.Now().Unix(),
				}
			}

			// Create a more unique block with randomized data to ensure different hashes
			newBlock := &blockchain.Block{
				Index:        nextHeight,
				Timestamp:    time.Now().Unix(),
				Data:         []byte(fmt.Sprintf("Block data for round %d - %d", nextHeight, rand.Int())),
				PrevHash:     prevBlock.Hash,
				Nonce:        rand.Uint64() % 1000, // Start with a random nonce
				StateRoot:    n.Chain.CurrentState.GetRootHash(),
				Transactions: transactions,
			}

			// Perform PoW to find a valid hash with real difficulty checking
			target := big.NewInt(1)
			target.Lsh(target, 256-24) // Using a fixed difficulty of 24

			for i := uint64(0); i < 100000; i++ {
				newBlock.Nonce = newBlock.Nonce + i
				hash := sha256.Sum256([]byte(fmt.Sprintf("%d%d%s%s%d",
					newBlock.Index, newBlock.Timestamp, newBlock.Data, newBlock.PrevHash, newBlock.Nonce)))

				// Check if hash meets difficulty
				var hashInt big.Int
				hashInt.SetBytes(hash[:])

				// If hash < target, we found a valid block
				if hashInt.Cmp(target) == -1 {
					newBlock.Hash = hex.EncodeToString(hash[:])
					break
				}
			}

			// Create and add the advanced block to our local chain
			advBlock := n.Chain.AddBlockWithNonce(newBlock.Data, newBlock.Transactions, newBlock.Nonce)
			log.Printf("[LEADER] Produced block at height=%d, hash=%s, nonce=%d",
				advBlock.Index, advBlock.Hash, advBlock.Nonce)

			// Broadcast advanced block
			msg, _ := json.Marshal(wireAdvancedBlock{Block: advBlock})
			_ = n.PubSub.Publish(BlocksTopic, msg)
			log.Printf("[BROADCAST] Block: height=%d hash=%s nonce=%d",
				advBlock.Index, advBlock.Hash, advBlock.Nonce)
		}
	}
}

// Advanced block consumption
func (n *Node) advancedBlockConsumer() {
	for {
		msg, err := n.BlockSub.Next(n.Ctx)
		if err != nil {
			return
		}

		// Check if the message is from this node
		fromOurself := msg.ReceivedFrom == n.Host.ID()

		// Try to decode as advanced block
		var wab wireAdvancedBlock
		if err := json.Unmarshal(msg.Data, &wab); err != nil || wab.Block == nil {
			// Try regular block format
			var wb wireBlock
			if err := json.Unmarshal(msg.Data, &wb); err != nil || wb.Block == nil {
				continue
			}

			// Skip further processing if the block is from ourselves
			if fromOurself {
				continue
			}

			// Process regular block
			log.Printf("[RECEIVED] Regular block from peer: height=%d hash=%s",
				wb.Block.Index, wb.Block.Hash)

			// Handle genesis block separately - important for synchronization
			if wb.Block.Index == 0 {
				log.Printf("[SYNC] Received genesis block with hash=%s", wb.Block.Hash)
				// Only replace genesis if we haven't added any blocks yet
				if len(n.Chain.Blocks) <= 1 {
					n.Chain.Blocks[0] = wb.Block
					log.Printf("[CHAIN] Updated genesis block to hash=%s", wb.Block.Hash)
				}
				continue
			}

			// Check if we should accept this block
			latestBlock := n.Chain.GetLatestBlock()

			// Apply block directly if it's the next height
			if wb.Block.Index == latestBlock.Index+1 && wb.Block.PrevHash == latestBlock.Hash {
				log.Printf("[SYNC] Adding block from peer at height %d", wb.Block.Index)
				// Add to chain directly
				n.Chain.Blocks = append(n.Chain.Blocks, wb.Block)
				// Update state with transactions
				for _, tx := range wb.Block.Transactions {
					n.Chain.CurrentState.ApplyTransaction(&tx)
				}
				log.Printf("[CHAIN] height=%d hash=%s",
					wb.Block.Index, wb.Block.Hash)
			} else if wb.Block.Index > latestBlock.Index+1 {
				// We're behind, log details for debugging
				log.Printf("[SYNC] Node is behind. Local height=%d, received height=%d, ourHash=%s, prevHash=%s",
					latestBlock.Index, wb.Block.Index, latestBlock.Hash, wb.Block.PrevHash)

				// Try to adopt this chain if it's longer
				if n.resolveFork(wb.Block) {
					// First, create temporary copies of the blocks we'll use to replace our chain
					// In a real implementation, we'd request all the missing blocks
					// But for simplicity, we'll create a new chain with just what we have

					// Get our genesis block
					genesisBlock := n.Chain.Blocks[0]

					// Create new chains
					newBlocks := []*blockchain.Block{genesisBlock, wb.Block}

					// Create equivalent advanced blocks
					advGenesis := n.Chain.AdvancedBlocks[0]
					advBlock := blockchain.NewAdvancedBlock(
						wb.Block.Index,
						wb.Block.Data,
						wb.Block.PrevHash,
						wb.Block.Transactions,
						n.Chain.CurrentState)
					newAdvBlocks := []*blockchain.AdvancedBlock{advGenesis, advBlock}

					// Attempt to replace our chain with this longer one
					if n.Chain.ReplaceChain(newBlocks, newAdvBlocks) {
						log.Printf("[FORK] Successfully replaced our chain with a longer chain from height %d to %d",
							latestBlock.Index, wb.Block.Index)
					} else {
						log.Printf("[FORK] Failed to replace chain - validation failed")
					}
				}
			} else if wb.Block.Index <= latestBlock.Index {
				// We already have blocks at this height, check for forks
				if n.resolveFork(wb.Block) {
					// Similar logic for chain replacement
					log.Printf("[FORK] Attempting to replace our chain with a longer valid chain")

					// Get our genesis block
					genesisBlock := n.Chain.Blocks[0]

					// Create new chains
					newBlocks := []*blockchain.Block{genesisBlock, wb.Block}

					// Create equivalent advanced blocks
					advGenesis := n.Chain.AdvancedBlocks[0]
					advBlock := blockchain.NewAdvancedBlock(
						wb.Block.Index,
						wb.Block.Data,
						wb.Block.PrevHash,
						wb.Block.Transactions,
						n.Chain.CurrentState)
					newAdvBlocks := []*blockchain.AdvancedBlock{advGenesis, advBlock}

					// Attempt to replace our chain
					if n.Chain.ReplaceChain(newBlocks, newAdvBlocks) {
						log.Printf("[FORK] Successfully replaced our chain with a valid chain")
					} else {
						log.Printf("[FORK] Failed to replace chain - validation failed")
					}
				} else {
					log.Printf("[CHAIN] Ignoring block at height %d (current height=%d)",
						wb.Block.Index, latestBlock.Index)
				}
			}
		} else {
			// Skip further processing if the block is from ourselves
			if fromOurself {
				log.Printf("[BLOCK] Ignoring own broadcast block: height=%d hash=%s",
					wab.Block.Index, wab.Block.Hash)
				continue
			}

			log.Printf("[RECEIVED] Block from peer: height=%d hash=%s",
				wab.Block.Index, wab.Block.Hash)

			// Handle genesis block separately
			if wab.Block.Index == 0 {
				log.Printf("[SYNC] Received genesis block with hash=%s", wab.Block.Hash)
				// Only replace genesis if we haven't added any blocks yet
				if len(n.Chain.Blocks) <= 1 && len(n.Chain.AdvancedBlocks) <= 1 {
					// Update both regular and advanced genesis blocks
					n.Chain.Blocks[0] = &wab.Block.Block
					n.Chain.AdvancedBlocks[0] = wab.Block
					log.Printf("[CHAIN] Updated genesis block to hash=%s", wab.Block.Hash)
				}
				continue
			}

			// Apply block directly if it's the next height
			latestBlock := n.Chain.GetLatestBlock()
			if wab.Block.Index == latestBlock.Index+1 && wab.Block.PrevHash == latestBlock.Hash {
				log.Printf("[SYNC] Adding block from peer at height %d", wab.Block.Index)
				// Add to chain directly using the received block
				n.Chain.Blocks = append(n.Chain.Blocks, &wab.Block.Block)
				n.Chain.AdvancedBlocks = append(n.Chain.AdvancedBlocks, wab.Block)
				// Update state with transactions
				for _, tx := range wab.Block.Transactions {
					n.Chain.CurrentState.ApplyTransaction(&tx)
				}
				log.Printf("[CHAIN] height=%d hash=%s",
					wab.Block.Index, wab.Block.Hash)
			} else if wab.Block.Index > latestBlock.Index+1 {
				// We're behind, log details for debugging
				log.Printf("[SYNC] Node is behind. Local height=%d, received height=%d, ourHash=%s, prevHash=%s",
					latestBlock.Index, wab.Block.Index, latestBlock.Hash, wab.Block.PrevHash)

				// Try to adopt this chain if it's longer
				if n.resolveFork(&wab.Block.Block) {
					// First, create temporary copies of the blocks we'll use to replace our chain
					// In a real implementation, we'd request all the missing blocks
					// But for simplicity, we'll create a new chain with just what we have

					// Get our genesis block
					genesisBlock := n.Chain.Blocks[0]

					// Create new chains
					newBlocks := []*blockchain.Block{genesisBlock, &wab.Block.Block}
					newAdvBlocks := []*blockchain.AdvancedBlock{n.Chain.AdvancedBlocks[0], wab.Block}

					// Attempt to replace our chain with this longer one
					if n.Chain.ReplaceChain(newBlocks, newAdvBlocks) {
						log.Printf("[FORK] Successfully replaced our chain with a longer chain from height %d to %d",
							latestBlock.Index, wab.Block.Index)
					} else {
						log.Printf("[FORK] Failed to replace chain - validation failed")
					}
				}
			} else if wab.Block.Index <= latestBlock.Index {
				// We already have blocks at this height, check for forks
				if n.resolveFork(&wab.Block.Block) {
					// Similar logic for chain replacement
					log.Printf("[FORK] Attempting to replace our chain with a longer valid chain")

					// Get our genesis block
					genesisBlock := n.Chain.Blocks[0]

					// Create new chains
					newBlocks := []*blockchain.Block{genesisBlock, &wab.Block.Block}
					newAdvBlocks := []*blockchain.AdvancedBlock{n.Chain.AdvancedBlocks[0], wab.Block}

					// Attempt to replace our chain
					if n.Chain.ReplaceChain(newBlocks, newAdvBlocks) {
						log.Printf("[FORK] Successfully replaced our chain with a valid chain")
					} else {
						log.Printf("[FORK] Failed to replace chain - validation failed")
					}
				} else {
					log.Printf("[CHAIN] Ignoring block at height %d (current height=%d)",
						wab.Block.Index, latestBlock.Index)
				}
			}
		}
	}
}

// Consumes blocks from the pub‑sub topic and appends to local chain
// once they pass consensus verification.
func (n *Node) blockConsumer() {
	for {
		msg, err := n.BlockSub.Next(n.Ctx)
		if err != nil {
			return
		}
		if msg.ReceivedFrom == n.Host.ID() {
			continue // ignore our own
		}

		var wb wireBlock
		if err := json.Unmarshal(msg.Data, &wb); err != nil || wb.Block == nil {
			continue
		}

		log.Printf("[RECEIVED] Block from peer: height=%d hash=%s",
			wb.Block.Index, wb.Block.Hash)

		// Check if we should accept this block
		latestBlock := n.Chain.GetLatestBlock()

		// Apply block directly if it's the next height
		if wb.Block.Index == latestBlock.Index+1 && wb.Block.PrevHash == latestBlock.Hash {
			log.Printf("[SYNC] Adding block from peer at height %d", wb.Block.Index)
			// Add to chain directly
			n.Chain.Blocks = append(n.Chain.Blocks, wb.Block)
			// Update state with transactions
			for _, tx := range wb.Block.Transactions {
				n.Chain.CurrentState.ApplyTransaction(&tx)
			}
			log.Printf("[CHAIN] height=%d hash=%s",
				wb.Block.Index, wb.Block.Hash)
		} else if wb.Block.Index > latestBlock.Index+1 {
			// We're behind, log details and try to adopt this chain
			log.Printf("[SYNC] Node is behind. Local height=%d, received height=%d",
				latestBlock.Index, wb.Block.Index)

			// Try to adopt this chain if it's longer
			if n.resolveFork(wb.Block) {
				// First, create temporary copies of the blocks we'll use to replace our chain
				// In a real implementation, we'd request all the missing blocks
				// But for simplicity, we'll create a new chain with just what we have

				// Get our genesis block
				genesisBlock := n.Chain.Blocks[0]

				// Create new chains with what we have
				newBlocks := []*blockchain.Block{genesisBlock, wb.Block}

				// Create equivalent advanced blocks
				advGenesis := n.Chain.AdvancedBlocks[0]
				advBlock := blockchain.NewAdvancedBlock(
					wb.Block.Index,
					wb.Block.Data,
					wb.Block.PrevHash,
					wb.Block.Transactions,
					n.Chain.CurrentState)
				newAdvBlocks := []*blockchain.AdvancedBlock{advGenesis, advBlock}

				// Attempt to replace our chain with this longer one
				if n.Chain.ReplaceChain(newBlocks, newAdvBlocks) {
					log.Printf("[FORK] Successfully replaced our chain with a longer chain from height %d to %d",
						latestBlock.Index, wb.Block.Index)
				} else {
					log.Printf("[FORK] Failed to replace chain - validation failed")
				}
			}
		} else if wb.Block.Index <= latestBlock.Index {
			// We already have blocks at this height, check for forks
			if n.resolveFork(wb.Block) {
				// Similar logic for chain replacement
				log.Printf("[FORK] Attempting to replace our chain with a longer valid chain")

				// Get our genesis block
				genesisBlock := n.Chain.Blocks[0]

				// Create new chains
				newBlocks := []*blockchain.Block{genesisBlock, wb.Block}

				// Create equivalent advanced blocks
				advGenesis := n.Chain.AdvancedBlocks[0]
				advBlock := blockchain.NewAdvancedBlock(
					wb.Block.Index,
					wb.Block.Data,
					wb.Block.PrevHash,
					wb.Block.Transactions,
					n.Chain.CurrentState)
				newAdvBlocks := []*blockchain.AdvancedBlock{advGenesis, advBlock}

				// Attempt to replace our chain
				if n.Chain.ReplaceChain(newBlocks, newAdvBlocks) {
					log.Printf("[FORK] Successfully replaced our chain with a valid chain")
				} else {
					log.Printf("[FORK] Failed to replace chain - validation failed")
				}
			} else {
				log.Printf("[CHAIN] Ignoring block at height %d (current height=%d)",
					wb.Block.Index, latestBlock.Index)
			}
		}
	}
}

// DemonstrateAdvancedFeatures demonstrates the advanced features of the blockchain
func (n *Node) DemonstrateAdvancedFeatures() {
	log.Println("[DEMO] Starting demonstration of advanced blockchain features")

	// 1. Zero-Knowledge Proofs
	log.Printf("[DEMO] Demonstrating Zero-Knowledge Proofs")
	n.Chain.DemonstrateZKProofs()

	// 2. Multi-Level Merkle Tree
	log.Printf("[DEMO] Demonstrating Multi-Level Merkle Tree Operations")
	// Create some test transactions
	transactions := []blockchain.Transaction{
		{ID: "tx1", From: "user1", To: "user2", Value: 100, Data: []byte("Test tx 1")},
		{ID: "tx2", From: "user2", To: "user3", Value: 50, Data: []byte("Test tx 2")},
		{ID: "tx3", From: "user3", To: "user1", Value: 75, Data: []byte("Test tx 3")},
	}

	// Create a test block with these transactions
	latestBlock := n.Chain.GetLatestBlock()
	advBlock := blockchain.NewAdvancedBlock(
		latestBlock.Index+1,
		[]byte("Demo block"),
		latestBlock.Hash,
		transactions,
		n.Chain.CurrentState)

	// Verify a transaction is included in the block
	included := advBlock.VerifyTransactionIncluded(transactions[1])
	log.Printf("[DEMO] Transaction verification: %v", included)

	// 3. BFT Consensus
	log.Printf("[DEMO] Demonstrating Byzantine Fault Tolerance Features")
	// Start a new round
	n.Consensus.StartNewRound()

	// Register a few validators
	n.Consensus.RegisterValidator("node1")
	n.Consensus.RegisterValidator("node2")
	n.Consensus.RegisterValidator("node3")

	// Set this node as validator
	n.Consensus.SetAsValidator(true)

	// 4. State Archiving
	log.Printf("[DEMO] Demonstrating State Archiving and Compression")
	// Archive current state
	n.Chain.StateArchiver.ArchiveCurrentState()

	// Create and compress state
	stateBlob, err := n.Chain.CompressAndArchiveState()
	if err != nil {
		log.Printf("[DEMO] Error compressing state: %v", err)
	} else {
		log.Printf("[DEMO] State compressed: root=%s size=%d bytes",
			stateBlob.RootHash, len(stateBlob.CompressedData))
	}

	// 5. Entropy Validation
	log.Printf("[DEMO] Demonstrating Entropy-Based Block Validation")
	entropyValid := n.Chain.VerifyBlockEntropy(advBlock)
	log.Printf("[DEMO] Block entropy validation: %v (score: %.2f)",
		entropyValid, advBlock.EntropyScore)

	log.Printf("[DEMO] Advanced feature demonstration complete")
}

// NewNetSync creates a new network synchronizer
func NewNetSync(node *Node) *NetSync {
	return &NetSync{
		node:           node,
		discoveryPeers: make(map[string]bool),
		lastSync:       time.Now(),
	}
}

// Start begins network synchronization
func (ns *NetSync) Start() {
	// Subscribe to peer discovery topic
	sub, err := ns.node.PubSub.Subscribe(PeerDiscoveryTopic)
	if err != nil {
		log.Printf("[ERROR] Failed to subscribe to peer discovery: %v", err)
		return
	}

	// Start peer discovery handler
	go ns.handlePeerDiscovery(sub)

	// Start periodic announce
	go ns.announceLoop()

	// Start periodic sync check
	go ns.syncLoop()
}

// handlePeerDiscovery processes peer discovery messages
func (ns *NetSync) handlePeerDiscovery(sub *pubsub.Subscription) {
	for {
		msg, err := sub.Next(ns.node.Ctx)
		if err != nil {
			return
		}

		// Skip our own messages
		if msg.ReceivedFrom == ns.node.Host.ID() {
			continue
		}

		// Process peer address
		peerAddr := string(msg.Data)
		log.Printf("[DISCOVERY] Received peer address: %s", peerAddr)

		// Try to connect to this peer if we're not already connected
		if _, exists := ns.discoveryPeers[peerAddr]; !exists {
			ns.discoveryPeers[peerAddr] = true

			// Parse multiaddr
			addr, err := ma.NewMultiaddr(peerAddr)
			if err != nil {
				log.Printf("[ERROR] Invalid peer address: %v", err)
				continue
			}

			// Extract peer info
			peerInfo, err := peer.AddrInfoFromP2pAddr(addr)
			if err != nil {
				log.Printf("[ERROR] Failed to parse peer info: %v", err)
				continue
			}

			// Connect to peer
			log.Printf("[DISCOVERY] Connecting to discovered peer: %s", peerAddr)
			if err := ns.node.Host.Connect(ns.node.Ctx, *peerInfo); err != nil {
				log.Printf("[ERROR] Failed to connect to discovered peer: %v", err)
			} else {
				log.Printf("[DISCOVERY] Successfully connected to peer: %s", peerInfo.ID.String()[:12])
			}
		}
	}
}

// announceLoop periodically announces our presence to the network
func (ns *NetSync) announceLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	// Announce immediately on start
	ns.announcePresence()

	for {
		select {
		case <-ticker.C:
			ns.announcePresence()
		case <-ns.node.Ctx.Done():
			return
		}
	}
}

// announcePresence broadcasts our address to the network
func (ns *NetSync) announcePresence() {
	// Build our address list
	var addrs []ma.Multiaddr
	for _, addr := range ns.node.Host.Addrs() {
		fullAddr := fmt.Sprintf("%s/p2p/%s", addr.String(), ns.node.Host.ID().String())
		maddr, err := ma.NewMultiaddr(fullAddr)
		if err == nil {
			addrs = append(addrs, maddr)
		}
	}

	// Pick best address to broadcast (usually the one with public IP)
	if len(addrs) > 0 {
		bestAddr := addrs[0].String()
		log.Printf("[DISCOVERY] Broadcasting our address: %s", bestAddr)

		// Publish to discovery topic
		err := ns.node.PubSub.Publish(PeerDiscoveryTopic, []byte(bestAddr))
		if err != nil {
			log.Printf("[ERROR] Failed to publish discovery message: %v", err)
		}
	}
}

// syncLoop periodically checks if we need to sync with other peers
func (ns *NetSync) syncLoop() {
	ticker := time.NewTicker(45 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ns.syncWithNetwork()
		case <-ns.node.Ctx.Done():
			return
		}
	}
}

// syncWithNetwork ensures we're in sync with the rest of the network
func (ns *NetSync) syncWithNetwork() {
	peerCount := len(ns.node.Host.Network().Peers())
	log.Printf("[SYNC] Checking sync status, connected to %d peers", peerCount)

	// If no peers, nothing to sync with
	if peerCount == 0 {
		return
	}

	// Broadcast our latest block to help others sync
	latestBlock := ns.node.Chain.GetLatestBlock()
	if latestBlock != nil {
		log.Printf("[SYNC] Broadcasting our latest block: height=%d hash=%s",
			latestBlock.Index, latestBlock.Hash)

		// Also broadcast genesis for peers that might need it
		if len(ns.node.Chain.Blocks) > 0 {
			genesisBlock := ns.node.Chain.Blocks[0]
			msg, _ := json.Marshal(wireBlock{Block: genesisBlock})
			ns.node.PubSub.Publish(BlocksTopic, msg)
		}

		// For regular blocks, broadcast as advanced blocks if available
		if len(ns.node.Chain.AdvancedBlocks) > 0 {
			latestAdvBlock := ns.node.Chain.GetLatestAdvancedBlock()
			msg, _ := json.Marshal(wireAdvancedBlock{Block: latestAdvBlock})
			ns.node.PubSub.Publish(BlocksTopic, msg)
		}
	}

	ns.lastSync = time.Now()
}

// resolveFork implements the longest chain rule to handle blockchain forks
func (n *Node) resolveFork(receivedBlock *blockchain.Block) bool {
	latestBlock := n.Chain.GetLatestBlock()

	// Case 1: Received block is at the same height as our latest block but with a different hash
	// This is a potential fork at the same height
	if receivedBlock.Index == latestBlock.Index && receivedBlock.Hash != latestBlock.Hash {
		log.Printf("[FORK] Detected blockchain fork at height %d", receivedBlock.Index)
		log.Printf("[FORK] Local hash: %s", latestBlock.Hash)
		log.Printf("[FORK] Remote hash: %s", receivedBlock.Hash)

		// In this case, we'll wait to see which chain grows longer
		log.Printf("[FORK] Waiting to see which chain grows longer")
		return false
	}

	// Case 2: Received block is from a longer chain
	// This means we need to replace our chain with the longer one
	if receivedBlock.Index > latestBlock.Index {
		log.Printf("[FORK] Detected longer chain (height %d vs our %d)",
			receivedBlock.Index, latestBlock.Index)

		// To properly implement chain replacement, we should reconstruct the entire chain
		// For this implementation, we'll request all blocks from this peer
		log.Printf("[FORK] Attempting to adopt the longer chain as per consensus rules")

		// In a real implementation, we'd need to:
		// 1. Request all missing blocks
		// 2. Validate the full chain
		// 3. Replace our chain

		// For this simplified version, return true to indicate we should adopt this chain
		return true
	}

	return false
}
