// pkg/consensus/authentication.go
package consensus

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"math/big"
	"sync"
	"time"
)

// NodeCredentials stores node authentication information
type NodeCredentials struct {
	NodeID        string
	PublicKey     *ecdsa.PublicKey
	LastSeen      time.Time
	TrustScore    float64
	Authenticated bool
}

// NodeAuthenticator manages node authentication
type NodeAuthenticator struct {
	privateKey      *ecdsa.PrivateKey
	publicKey       *ecdsa.PublicKey
	knownNodes      map[string]*NodeCredentials
	challengeNonces map[string][]byte
	mutex           sync.RWMutex
	trustDecayRate  float64
}

// NewNodeAuthenticator creates a new node authenticator
func NewNodeAuthenticator() (*NodeAuthenticator, error) {
	// Generate key pair
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}

	return &NodeAuthenticator{
		privateKey:      privateKey,
		publicKey:       &privateKey.PublicKey,
		knownNodes:      make(map[string]*NodeCredentials),
		challengeNonces: make(map[string][]byte),
		trustDecayRate:  0.01, // 1% decay per check
	}, nil
}

// RegisterNode registers a new node
func (na *NodeAuthenticator) RegisterNode(nodeID string, publicKey *ecdsa.PublicKey) {
	na.mutex.Lock()
	defer na.mutex.Unlock()

	na.knownNodes[nodeID] = &NodeCredentials{
		NodeID:        nodeID,
		PublicKey:     publicKey,
		LastSeen:      time.Now(),
		TrustScore:    0.5, // Initial neutral trust
		Authenticated: false,
	}
}

// GenerateChallenge generates an authentication challenge for a node
func (na *NodeAuthenticator) GenerateChallenge(nodeID string) []byte {
	na.mutex.Lock()
	defer na.mutex.Unlock()

	// Generate random challenge
	nonce := make([]byte, 32)
	rand.Read(nonce)

	// Store nonce for verification
	na.challengeNonces[nodeID] = nonce

	return nonce
}

// VerifyResponse verifies a node's response to a challenge
func (na *NodeAuthenticator) VerifyResponse(nodeID string, signature []byte) bool {
	na.mutex.Lock()
	defer na.mutex.Unlock()

	node, exists := na.knownNodes[nodeID]
	if !exists {
		return false
	}

	nonce, exists := na.challengeNonces[nodeID]
	if !exists {
		return false
	}

	// Verify signature
	hash := sha256.Sum256(nonce)
	r := new(big.Int).SetBytes(signature[:32])
	s := new(big.Int).SetBytes(signature[32:])

	verified := ecdsa.Verify(node.PublicKey, hash[:], r, s)

	if verified {
		// Update node status
		node.LastSeen = time.Now()
		node.TrustScore += 0.05 // Increase trust on successful auth
		if node.TrustScore > 1.0 {
			node.TrustScore = 1.0
		}
		node.Authenticated = true

		// Remove used nonce
		delete(na.challengeNonces, nodeID)
	} else {
		// Penalize failed auth
		node.TrustScore -= 0.2
		if node.TrustScore < 0.0 {
			node.TrustScore = 0.0
		}
	}

	return verified
}

// SignMessage signs a message with this node's private key
func (na *NodeAuthenticator) SignMessage(message []byte) ([]byte, error) {
	hash := sha256.Sum256(message)
	r, s, err := ecdsa.Sign(rand.Reader, na.privateKey, hash[:])
	if err != nil {
		return nil, err
	}

	signature := append(r.Bytes(), s.Bytes()...)
	return signature, nil
}

// CheckNodeTrust checks if a node meets trust requirements
func (na *NodeAuthenticator) CheckNodeTrust(nodeID string, requiredTrust float64) bool {
	na.mutex.RLock()
	defer na.mutex.RUnlock()

	node, exists := na.knownNodes[nodeID]
	if !exists {
		return false
	}

	// Apply time-based trust decay
	na.applyTrustDecay(node)

	return node.Authenticated && node.TrustScore >= requiredTrust
}

// applyTrustDecay applies time-based decay to trust scores
func (na *NodeAuthenticator) applyTrustDecay(node *NodeCredentials) {
	// Calculate time since last seen
	timeSinceLastSeen := time.Since(node.LastSeen)

	// Apply decay based on hours passed
	hoursPassed := timeSinceLastSeen.Hours()
	decayFactor := 1.0 - (na.trustDecayRate * hoursPassed)

	if decayFactor < 0.0 {
		decayFactor = 0.0
	}

	node.TrustScore *= decayFactor

	// If too much time has passed, deauthenticate
	if timeSinceLastSeen > 24*time.Hour {
		node.Authenticated = false
	}
}
