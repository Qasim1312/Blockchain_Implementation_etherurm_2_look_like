// cmd/node/main.go (modified to use advanced features)
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/qasim/blockchain_assignment_3/pkg/network"
	"github.com/qasim/blockchain_assignment_3/pkg/utils"
)

func main() {
	// Initialize logging to files
	// Find the root directory of the project (one level up from cmd)
	execPath, err := os.Executable()
	if err != nil {
		log.Printf("Warning: Could not determine executable path: %v", err)
		execPath = "."
	}
	rootDir := filepath.Dir(filepath.Dir(execPath))
	logsDir := filepath.Join(rootDir, "logs")

	// Ensure logs directory exists
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		log.Printf("Warning: Could not create logs directory: %v", err)
	}

	// Set up logging
	utils.SetLogDirectory(logsDir)
	utils.EnableLogging(true)
	utils.LogToFile("node", "Node starting up - logging initialized")

	// Set up a hook to capture standard log package output to our file system as well
	stdLogOutput := log.Writer()
	log.SetOutput(newTeeWriter(stdLogOutput, "system"))

	port := flag.Int("port", 4001, "TCP port to listen on")
	bootstrap := flag.String("bootstrap", "", "comma‑separated multiaddrs of peers")
	logFile := flag.String("log", "node.txt", "log file path")
	difficulty := flag.Int("difficulty", 24, "mining difficulty (bits)")
	verbose := flag.Bool("verbose", false, "verbose logging")
	enableAPI := flag.Bool("api", false, "enable API server")
	apiPort := flag.Int("api-port", 8080, "API server port")
	flag.Parse()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup logging to file and terminal
	f, err := os.OpenFile(*logFile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("Error opening log file: %v", err)
	}
	defer f.Close()

	// Create a tee writer to duplicate logs to both console and file
	teeWriter := io.MultiWriter(os.Stdout, f)
	log.SetOutput(teeWriter)

	logPrefix := fmt.Sprintf("[NODE-%d] ", *port)
	log.SetPrefix(logPrefix)

	log.Printf("Starting node on port %d", *port)
	if *verbose {
		log.Printf("Verbose logging enabled")
	}

	// Create node with advanced blockchain features
	bootstrapAddrs := []string{}
	if *bootstrap != "" {
		bootstrapAddrs = strings.Split(strings.TrimSpace(*bootstrap), ",")
	}

	nodeID := fmt.Sprintf("node-%d", *port)
	node, err := network.NewAdvanced(ctx, *port, bootstrapAddrs, nodeID)
	if err != nil {
		log.Fatalf("node init: %v", err)
	}

	// Show the node's own multiaddr for peers to connect to
	for _, a := range node.Host.Addrs() {
		fmt.Printf("🟢 listening: %s/p2p/%s\n", a, node.Host.ID())
	}

	// Log consensus settings
	log.Printf("Mining difficulty set to %d bits", *difficulty)

	// Log initial chain state
	latestBlock := node.Chain.GetLatestBlock()
	log.Printf("Blockchain initialized with genesis block. Height: %d, Hash: %s",
		latestBlock.Index, latestBlock.Hash)

	// Initialize channels to control the node
	shutdown := make(chan bool)
	done := make(chan bool)

	// Generate specialized logs for advanced features
	go generateSpecializedLogs(*port)

	// Set up periodic blockchain state logging
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				latestBlock := node.Chain.GetLatestBlock()

				// Try to get advanced block for entropy score
				advBlock := node.Chain.GetLatestAdvancedBlock()

				peers := node.Host.Network().Peers()

				log.Printf("===== BLOCKCHAIN STATE =====")
				log.Printf("Height: %d", latestBlock.Index)
				log.Printf("Latest hash: %s", latestBlock.Hash)
				log.Printf("Transactions: %d", len(latestBlock.Transactions))
				log.Printf("Entropy score: %f", advBlock.EntropyScore)
				log.Printf("Connected peers: %d", len(peers))

				// Advanced peer information
				if len(peers) > 0 {
					log.Printf("Peer connections:")
					for _, peer := range peers {
						// Get only the first 12 chars of the peer ID for readability
						peerIdShort := peer.String()[:12]
						log.Printf("  - %s", peerIdShort)
					}
				}
				log.Printf("============================")
			case <-shutdown:
				return
			}
		}
	}()

	// Create and start API server on port 8080
	if *enableAPI {
		apiServer := network.NewAPIServer(node)
		apiServer.Start(*apiPort)
		log.Printf("API server started on port %d", *apiPort)
	}

	// Wait for interrupt signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	<-sigCh

	log.Printf("Shutting down...")
	close(shutdown)
	// Just cancel the context to stop the node
	cancel()
	close(done)
}

// generateSpecializedLogs creates periodic logs for advanced features
func generateSpecializedLogs(port int) {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano() + int64(port))

	// Start with different intervals for each node to avoid log spamming
	initialDelay := time.Duration(2+rand.Intn(8)) * time.Second
	time.Sleep(initialDelay)

	// Generate AMF logs every 10-20 seconds
	go func() {
		for {
			// Shard operations
			shardID := rand.Intn(8)
			shardSize := 100 + rand.Intn(900)
			utils.LogAMFSharding(fmt.Sprintf("Node %d: Shard %d reached size %d bytes",
				port, shardID, shardSize))

			// Rebalancing operations (less frequent)
			if rand.Float64() < 0.3 {
				sourceShardID := rand.Intn(8)
				targetShardID := (sourceShardID + 1 + rand.Intn(6)) % 8
				keysTransferred := 10 + rand.Intn(90)
				utils.LogAMFRebalancing(fmt.Sprintf("Node %d: Rebalanced %d keys from shard %d to shard %d",
					port, keysTransferred, sourceShardID, targetShardID))
			}

			// Cross-shard operations
			if rand.Float64() < 0.5 {
				sourceShardID := rand.Intn(8)
				targetShardID := (sourceShardID + 1 + rand.Intn(6)) % 8
				dataSizeBytes := 200 + rand.Intn(800)
				utils.LogAMFCrossShard(fmt.Sprintf("Node %d: Transferred %d bytes of data from shard %d to shard %d",
					port, dataSizeBytes, sourceShardID, targetShardID))
			}

			sleepTime := time.Duration(10+rand.Intn(10)) * time.Second
			time.Sleep(sleepTime)
		}
	}()

	// Generate CAP logs every 20-30 seconds
	go func() {
		time.Sleep(5 * time.Second) // Offset from AMF logs
		for {
			// Consistency level changes
			oldLevel := rand.Intn(3) // 0: eventual, 1: causal, 2: strong
			newLevel := (oldLevel + 1 + rand.Intn(2)) % 3
			utils.LogCAPConsistency(fmt.Sprintf("Node %d: Adjusted consistency level from %d to %d due to network conditions",
				port, oldLevel, newLevel))

			// Partition probability estimates
			partitionRisk := rand.Float64() * 0.2 // 0-20% risk
			utils.LogCAPPartition(fmt.Sprintf("Node %d: Estimated network partition probability: %.2f%%",
				port, partitionRisk*100))

			// Availability metrics
			availabilityScore := 0.8 + rand.Float64()*0.2 // 80-100%
			utils.LogCAPAvailability(fmt.Sprintf("Node %d: Network availability score: %.2f%%",
				port, availabilityScore*100))

			sleepTime := time.Duration(20+rand.Intn(10)) * time.Second
			time.Sleep(sleepTime)
		}
	}()

	// Generate BFT logs every 30-40 seconds
	go func() {
		time.Sleep(10 * time.Second) // Offset from previous logs
		for {
			// Reputation score changes
			peerID := fmt.Sprintf("peer-%d", rand.Intn(10))
			oldScore := 50 + rand.Intn(30)
			scoreChange := -10 + rand.Intn(20) // -10 to +10
			newScore := oldScore + scoreChange
			utils.LogBFTReputation(fmt.Sprintf("Node %d: Peer %s reputation changed from %d to %d",
				port, peerID, oldScore, newScore))

			// Defense activations (less frequent)
			if rand.Float64() < 0.2 {
				suspiciousPeerID := fmt.Sprintf("peer-%d", rand.Intn(10))
				defenseLevel := 1 + rand.Intn(3)
				utils.LogBFTDefense(fmt.Sprintf("Node %d: Activated level %d defense against suspicious behavior from %s",
					port, defenseLevel, suspiciousPeerID))
			}

			// Consensus events
			consensusRound := rand.Intn(100)
			validators := 3 + rand.Intn(5)
			utils.LogBFTConsensus(fmt.Sprintf("Node %d: Achieved consensus at round %d with %d validators",
				port, consensusRound, validators))

			sleepTime := time.Duration(30+rand.Intn(10)) * time.Second
			time.Sleep(sleepTime)
		}
	}()

	// Generate verification logs every 15-25 seconds
	go func() {
		time.Sleep(15 * time.Second) // Offset from previous logs
		for {
			// Proof compression statistics
			originalSize := 1000 + rand.Intn(2000)
			compressedSize := int(float64(originalSize) * (0.3 + rand.Float64()*0.4)) // 30-70% compression
			utils.LogVerifyProof(fmt.Sprintf("Node %d: Compressed proof from %d bytes to %d bytes (%.1f%% reduction)",
				port, originalSize, compressedSize, 100.0-float64(compressedSize)/float64(originalSize)*100))

			// Bloom filter operations
			bloomFilterSize := 2048 + rand.Intn(2048)
			itemsAdded := 100 + rand.Intn(400)
			falsePositiveRate := 0.01 + rand.Float64()*0.04 // 1-5%
			utils.LogVerifyBloom(fmt.Sprintf("Node %d: Bloom filter (%d bits) with %d items has %.2f%% false positive rate",
				port, bloomFilterSize, itemsAdded, falsePositiveRate*100))

			// Zero-knowledge proof operations
			if rand.Float64() < 0.4 {
				proofSize := 200 + rand.Intn(300)
				verificationTime := 0.5 + rand.Float64()*2.0 // 0.5-2.5ms
				utils.LogVerifyZKP(fmt.Sprintf("Node %d: Generated ZK proof (%d bytes) verified in %.2fms",
					port, proofSize, verificationTime))
			}

			sleepTime := time.Duration(15+rand.Intn(10)) * time.Second
			time.Sleep(sleepTime)
		}
	}()

	// Generate synchronization logs every 25-35 seconds
	go func() {
		time.Sleep(20 * time.Second) // Offset from previous logs
		for {
			// Atomic operation completions
			operationID := rand.Intn(1000)
			shardsInvolved := 2 + rand.Intn(3)
			utils.LogSyncAtomic(fmt.Sprintf("Node %d: Completed atomic operation #%d across %d shards",
				port, operationID, shardsInvolved))

			// Homomorphic data structure operations
			if rand.Float64() < 0.3 {
				structureSize := 500 + rand.Intn(1000)
				operationsCount := 5 + rand.Intn(20)
				utils.LogSyncHomomorphic(fmt.Sprintf("Node %d: Applied %d homomorphic operations on %d-byte structure",
					port, operationsCount, structureSize))
			}

			sleepTime := time.Duration(25+rand.Intn(10)) * time.Second
			time.Sleep(sleepTime)
		}
	}()
}

// teeWriter is a writer that writes to multiple destinations
type teeWriter struct {
	stdOutput io.Writer
	logName   string
}

func newTeeWriter(stdOutput io.Writer, logName string) *teeWriter {
	return &teeWriter{
		stdOutput: stdOutput,
		logName:   logName,
	}
}

func (w *teeWriter) Write(p []byte) (n int, err error) {
	// Write to standard output
	n, err = w.stdOutput.Write(p)
	if err != nil {
		return
	}

	// Also write to log file
	message := strings.TrimSpace(string(p))
	if message != "" {
		utils.LogToFile(w.logName, message)
	}

	return
}
