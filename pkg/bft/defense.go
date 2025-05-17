// pkg/bft/defense.go
package bft

import (
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/qasim/blockchain_assignment_3/pkg/utils"
)

// ReputationScore represents a node's trustworthiness
type ReputationScore struct {
	TotalInteractions  uint64
	SuccessfulActions  uint64
	FailedActions      uint64
	SuspiciousPatterns uint64
	LastUpdateTime     time.Time
	HistoricalScores   []float64
	ScoringTimes       []time.Time
	TrendDirection     float64 // Positive means improving, negative means declining
	StandardDeviation  float64
	ConsistencyScore   float64 // How consistent the node behavior is
	AnomalyCount       int     // Count of detected anomalies
}

// NodeReputation manages reputation for nodes in the network
type NodeReputation struct {
	scores             map[string]*ReputationScore
	thresholds         ReputationThresholds
	mutex              sync.RWMutex
	decayPeriod        time.Duration
	decayFactor        float64
	adaptiveThresholds bool
	networkHealth      float64 // 0.0-1.0 indicating overall network health
	adaptationHistory  []ThresholdAdjustment
	performanceModels  map[string]*PerformanceModel
}

// ReputationThresholds defines acceptable behavior thresholds
type ReputationThresholds struct {
	MinReputationScore float64
	WarningThreshold   float64
	BanThreshold       float64
	ProbationaryPeriod time.Duration
	RecoveryRate       float64 // How quickly nodes can recover reputation
	DecayRate          float64 // How quickly reputation decays with inactivity
}

// NewNodeReputation creates a new node reputation system
func NewNodeReputation() *NodeReputation {
	return &NodeReputation{
		scores: make(map[string]*ReputationScore),
		thresholds: ReputationThresholds{
			MinReputationScore: 0.3,
			WarningThreshold:   0.5,
			BanThreshold:       0.2,
			ProbationaryPeriod: 6 * time.Hour,
			RecoveryRate:       0.05, // 5% recovery rate
			DecayRate:          0.01, // 1% decay rate per period
		},
		decayPeriod:        24 * time.Hour,
		decayFactor:        0.95, // 5% decay per period
		adaptiveThresholds: true,
		networkHealth:      0.8, // Start with assumption of good health
		adaptationHistory:  make([]ThresholdAdjustment, 0),
		performanceModels:  make(map[string]*PerformanceModel),
	}
}

// ReportSuccess reports a successful interaction with a node
func (nr *NodeReputation) ReportSuccess(nodeID string) {
	nr.mutex.Lock()
	defer nr.mutex.Unlock()

	score, exists := nr.scores[nodeID]
	if !exists {
		score = &ReputationScore{
			LastUpdateTime:   time.Now(),
			HistoricalScores: make([]float64, 0, 10),
			ScoringTimes:     make([]time.Time, 0, 10),
		}
		nr.scores[nodeID] = score
	}

	// Save old score for comparison
	oldScore := nr.calculateNodeScore(score)

	// Update scores
	score.TotalInteractions++
	score.SuccessfulActions++
	score.LastUpdateTime = time.Now()

	// Calculate and store current score for history
	currentScore := nr.calculateNodeScore(score)

	// Store historical score
	score.HistoricalScores = append(score.HistoricalScores, currentScore)
	score.ScoringTimes = append(score.ScoringTimes, time.Now())

	// Limit history size
	maxHistory := 20
	if len(score.HistoricalScores) > maxHistory {
		score.HistoricalScores = score.HistoricalScores[len(score.HistoricalScores)-maxHistory:]
		score.ScoringTimes = score.ScoringTimes[len(score.ScoringTimes)-maxHistory:]
	}

	// Update trend direction if we have enough history
	if len(score.HistoricalScores) >= 3 {
		nr.updateTrendDirection(score)
	}

	// Update performance model
	nr.updatePerformanceModel(nodeID, currentScore)

	// Log significant changes (more than 5%)
	if math.Abs(currentScore-oldScore) > 0.05 {
		utils.LogToFile("bft_reputation", fmt.Sprintf("Node %s reputation increased from %.3f to %.3f (successes: %d, failures: %d)",
			nodeID, oldScore, score.SuccessfulActions, score.FailedActions))
	}
}

// ReportFailure reports a failed interaction with a node
func (nr *NodeReputation) ReportFailure(nodeID string) {
	nr.mutex.Lock()
	defer nr.mutex.Unlock()

	score, exists := nr.scores[nodeID]
	if !exists {
		score = &ReputationScore{
			LastUpdateTime:   time.Now(),
			HistoricalScores: make([]float64, 0, 10),
			ScoringTimes:     make([]time.Time, 0, 10),
		}
		nr.scores[nodeID] = score
	}

	// Save old score for comparison
	oldScore := nr.calculateNodeScore(score)

	// Update scores
	score.TotalInteractions++
	score.FailedActions++
	score.LastUpdateTime = time.Now()

	// Calculate and store current score for history
	currentScore := nr.calculateNodeScore(score)

	// Store historical score
	score.HistoricalScores = append(score.HistoricalScores, currentScore)
	score.ScoringTimes = append(score.ScoringTimes, time.Now())

	// Limit history size
	maxHistory := 20
	if len(score.HistoricalScores) > maxHistory {
		score.HistoricalScores = score.HistoricalScores[len(score.HistoricalScores)-maxHistory:]
		score.ScoringTimes = score.ScoringTimes[len(score.ScoringTimes)-maxHistory:]
	}

	// Update trend direction if we have enough history
	if len(score.HistoricalScores) >= 3 {
		nr.updateTrendDirection(score)
	}

	// Update performance model
	nr.updatePerformanceModel(nodeID, currentScore)

	// Check if node has dropped below critical threshold
	if currentScore < 0.2 && oldScore >= 0.2 {
		utils.LogToFile("bft_defense", fmt.Sprintf("Defense activated against node %s (reputation dropped to %.3f, critical threshold breached)",
			nodeID, currentScore))
	}

	// Log significant changes (more than 5%)
	if math.Abs(currentScore-oldScore) > 0.05 {
		utils.LogToFile("bft_reputation", fmt.Sprintf("Node %s reputation decreased from %.3f to %.3f (successes: %d, failures: %d)",
			nodeID, oldScore, currentScore, score.SuccessfulActions, score.FailedActions))
	}
}

// ReportSuspiciousActivity reports suspicious activity from a node
func (nr *NodeReputation) ReportSuspiciousActivity(nodeID string) {
	nr.mutex.Lock()
	defer nr.mutex.Unlock()

	score, exists := nr.scores[nodeID]
	if !exists {
		score = &ReputationScore{
			LastUpdateTime: time.Now(),
		}
		nr.scores[nodeID] = score
	}

	// Apply time decay
	nr.applyDecay(score)

	// Update scores - suspicious activity has higher impact
	score.TotalInteractions++
	score.SuspiciousPatterns++
	score.LastUpdateTime = time.Now()
}

// GetReputationScore calculates the reputation score for a node
func (nr *NodeReputation) GetReputationScore(nodeID string) float64 {
	nr.mutex.RLock()
	defer nr.mutex.RUnlock()

	score, exists := nr.scores[nodeID]
	if !exists {
		return 0.5 // Default score for unknown nodes
	}

	// Apply time decay
	nr.applyDecay(score)

	// Calculate score based on history
	if score.TotalInteractions == 0 {
		return 0.5 // No interactions yet
	}

	// Base score from successful vs. failed actions
	baseScore := float64(score.SuccessfulActions) / float64(score.TotalInteractions)

	// Apply penalty for suspicious patterns
	suspiciousPenalty := math.Min(float64(score.SuspiciousPatterns)/float64(score.TotalInteractions)*2.0, 0.8)

	finalScore := baseScore * (1.0 - suspiciousPenalty)

	return finalScore
}

// IsNodeTrusted checks if a node meets trust requirements
func (nr *NodeReputation) IsNodeTrusted(nodeID string) bool {
	score := nr.GetReputationScore(nodeID)
	return score >= nr.thresholds.MinReputationScore
}

// ShouldBanNode checks if a node should be banned
func (nr *NodeReputation) ShouldBanNode(nodeID string) bool {
	score := nr.GetReputationScore(nodeID)
	return score < nr.thresholds.BanThreshold
}

// applyDecay applies time-based decay to reputation scores
func (nr *NodeReputation) applyDecay(score *ReputationScore) {
	timeSinceUpdate := time.Since(score.LastUpdateTime)
	if timeSinceUpdate < nr.decayPeriod {
		return // Not enough time has passed
	}

	// Calculate decay factor based on elapsed time
	decayPeriods := float64(timeSinceUpdate) / float64(nr.decayPeriod)
	cumulativeDecay := math.Pow(nr.decayFactor, decayPeriods)

	// Apply decay to all metrics
	score.SuccessfulActions = uint64(float64(score.SuccessfulActions) * cumulativeDecay)
	score.FailedActions = uint64(float64(score.FailedActions) * cumulativeDecay)
	score.SuspiciousPatterns = uint64(float64(score.SuspiciousPatterns) * cumulativeDecay)
	score.TotalInteractions = score.SuccessfulActions + score.FailedActions + score.SuspiciousPatterns

	// Update last update time
	score.LastUpdateTime = time.Now()
}

// AnomalyDetector implements anomaly detection for Byzantine behavior
type AnomalyDetector struct {
	observations    map[string][]float64
	thresholds      map[string]float64
	detectionWindow int
	mutex           sync.RWMutex
}

// CryptographicDefense implements cryptographic protection mechanisms
type CryptographicDefense struct {
	validatorKeys map[string][]byte
	signatures    map[string][]byte
	threshold     int
	mutex         sync.RWMutex
}

// MultilayerDefense provides a multi-layer defensive framework for the consensus
type MultilayerDefense struct {
	nodeReputations map[string]float64
	threatScores    map[string]float64
	anomalyDetector *AnomalyDetector
	cryptoDefense   *CryptographicDefense
	mutex           sync.RWMutex
}

// NewMultilayerDefense creates a new BFT defense system
func NewMultilayerDefense() *MultilayerDefense {
	anomalyDetector := &AnomalyDetector{
		observations:    make(map[string][]float64),
		thresholds:      make(map[string]float64),
		detectionWindow: 100, // Monitor last 100 observations
	}

	cryptoDefense := &CryptographicDefense{
		validatorKeys: make(map[string][]byte),
		signatures:    make(map[string][]byte),
		threshold:     3, // Require at least 3 valid signatures
	}

	return &MultilayerDefense{
		nodeReputations: make(map[string]float64),
		threatScores:    make(map[string]float64),
		anomalyDetector: anomalyDetector,
		cryptoDefense:   cryptoDefense,
	}
}

// UpdateNodeReputation updates a node's reputation based on behavior
func (md *MultilayerDefense) UpdateNodeReputation(nodeID string, behavior float64) {
	md.mutex.Lock()
	defer md.mutex.Unlock()

	currentRep, exists := md.nodeReputations[nodeID]
	if !exists {
		// Initialize new nodes with neutral reputation
		currentRep = 0.5
	}

	// Update reputation with dampening to prevent wild swings
	// behavior should be between 0 (bad) and 1 (good)
	newRep := currentRep*0.9 + behavior*0.1

	// Keep reputation between 0 and 1
	newRep = math.Max(0, math.Min(1, newRep))

	md.nodeReputations[nodeID] = newRep

	log.Printf("Updated node %s reputation: %.6f -> %.6f", nodeID, currentRep, newRep)
}

// GetNodeReputation returns a node's current reputation
func (md *MultilayerDefense) GetNodeReputation(nodeID string) float64 {
	md.mutex.RLock()
	defer md.mutex.RUnlock()

	rep, exists := md.nodeReputations[nodeID]
	if !exists {
		return 0.5 // Default neutral reputation
	}
	return rep
}

// UpdateThreatScore updates a node's threat score based on suspicious activity
func (md *MultilayerDefense) UpdateThreatScore(nodeID string, suspiciousActivity float64) {
	md.mutex.Lock()
	defer md.mutex.Unlock()

	currentScore, exists := md.threatScores[nodeID]
	if !exists {
		currentScore = 0
	}

	// Increase threat score based on suspicious activity
	// Activity should be between 0 (normal) and 1 (highly suspicious)
	newScore := currentScore*0.95 + suspiciousActivity*0.05

	// Apply node reputation as a dampening factor
	nodeRep := md.GetNodeReputation(nodeID)
	dampening := 1.0 - nodeRep // Higher reputation = more dampening of threats

	newScore = newScore * dampening

	// Keep score between 0 and 1
	newScore = math.Max(0, math.Min(1, newScore))

	md.threatScores[nodeID] = newScore

	if newScore > 0.7 {
		log.Printf("WARNING: Node %s has high threat score: %.6f", nodeID, newScore)
	}
}

// IsNodeTrusted determines if a node's threat score is below threshold
func (md *MultilayerDefense) IsNodeTrusted(nodeID string) bool {
	md.mutex.RLock()
	defer md.mutex.RUnlock()

	score, exists := md.threatScores[nodeID]
	if !exists {
		return true // Default trusted
	}

	// Nodes with threat scores above 0.7 are not trusted
	return score < 0.7
}

// RecordObservation records a behavioral observation for anomaly detection
func (ad *AnomalyDetector) RecordObservation(metric string, value float64) {
	ad.mutex.Lock()
	defer ad.mutex.Unlock()

	observations, exists := ad.observations[metric]
	if !exists {
		observations = make([]float64, 0, ad.detectionWindow)
	}

	// Add new observation
	observations = append(observations, value)

	// Keep only the most recent observations
	if len(observations) > ad.detectionWindow {
		observations = observations[len(observations)-ad.detectionWindow:]
	}

	ad.observations[metric] = observations

	// Update threshold if we have enough data
	if len(observations) >= 10 {
		mean, stdDev := calculateStats(observations)
		// Set threshold at 3 standard deviations from mean
		ad.thresholds[metric] = mean + 3*stdDev
	}
}

// DetectAnomaly checks if a value is anomalous based on recorded observations
func (ad *AnomalyDetector) DetectAnomaly(metric string, value float64) bool {
	ad.mutex.RLock()
	defer ad.mutex.RUnlock()

	threshold, exists := ad.thresholds[metric]
	if !exists {
		return false // No threshold yet, so can't detect anomaly
	}

	return value > threshold
}

// calculateStats calculates mean and standard deviation of a slice of values
func calculateStats(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}

	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	variance := sumSquaredDiff / float64(len(values))
	stdDev := math.Sqrt(variance)

	return mean, stdDev
}

// RegisterValidator registers a validator's key for signature verification
func (cd *CryptographicDefense) RegisterValidator(nodeID string, publicKey []byte) {
	cd.mutex.Lock()
	defer cd.mutex.Unlock()

	cd.validatorKeys[nodeID] = publicKey
	log.Printf("Registered validator %s with public key", nodeID)
}

// VerifySignature verifies a signature against a registered validator key
func (cd *CryptographicDefense) VerifySignature(nodeID string, message, signature []byte) bool {
	cd.mutex.RLock()
	defer cd.mutex.RUnlock()

	publicKey, exists := cd.validatorKeys[nodeID]
	if !exists {
		return false // No key registered for this validator
	}

	// In a real implementation, this would use proper cryptographic verification
	// Here we'll just check if the signature matches the one we've seen before
	prevSig, exists := cd.signatures[nodeID]
	if exists {
		// Check if this is a duplicate signature (potential replay attack)
		sameSignature := len(signature) == len(prevSig)
		if sameSignature {
			for i := 0; i < len(signature); i++ {
				if signature[i] != prevSig[i] {
					sameSignature = false
					break
				}
			}
		}

		if sameSignature {
			log.Printf("WARNING: Potential replay attack from node %s", nodeID)
			return false
		}
	}

	// Store the signature for future checks
	cd.signatures[nodeID] = signature

	// Simulate verification (in real implementation would use crypto library)
	return len(publicKey) > 0 && len(signature) > 0
}

// GetThresholdStatus returns whether enough valid signatures have been collected
func (cd *CryptographicDefense) GetThresholdStatus() (int, bool) {
	cd.mutex.RLock()
	defer cd.mutex.RUnlock()

	validSigs := len(cd.signatures)
	return validSigs, validSigs >= cd.threshold
}

// Add new types for adaptive threshold tracking
type ThresholdAdjustment struct {
	Timestamp          time.Time
	PreviousThresholds ReputationThresholds
	NewThresholds      ReputationThresholds
	Reason             string
	AffectedNodeCount  int
}

type PerformanceModel struct {
	NodeID           string
	ExpectedScore    float64
	Variance         float64
	ConfidenceLevel  float64
	LastUpdated      time.Time
	PredictionErrors []float64
}

// EnableAdaptiveThresholds enables or disables adaptive threshold adjustment
func (nr *NodeReputation) EnableAdaptiveThresholds(enabled bool) {
	nr.mutex.Lock()
	defer nr.mutex.Unlock()
	nr.adaptiveThresholds = enabled
}

// UpdateNetworkHealth updates the overall network health metric
func (nr *NodeReputation) UpdateNetworkHealth(health float64) {
	nr.mutex.Lock()
	defer nr.mutex.Unlock()

	// Keep health in valid range
	if health < 0.0 {
		health = 0.0
	} else if health > 1.0 {
		health = 1.0
	}

	// If health changed significantly, adjust thresholds
	if math.Abs(nr.networkHealth-health) > 0.2 && nr.adaptiveThresholds {
		oldThresholds := nr.thresholds

		// Adjust thresholds based on health
		if health < 0.5 {
			// Network in trouble, be more forgiving
			nr.thresholds.BanThreshold = math.Max(0.1, oldThresholds.BanThreshold*0.8)
			nr.thresholds.MinReputationScore = math.Max(0.2, oldThresholds.MinReputationScore*0.9)
		} else {
			// Network healthy, can be stricter
			nr.thresholds.BanThreshold = math.Min(0.25, oldThresholds.BanThreshold*1.1)
			nr.thresholds.MinReputationScore = math.Min(0.4, oldThresholds.MinReputationScore*1.1)
		}

		// Record adjustment
		adjustment := ThresholdAdjustment{
			Timestamp:          time.Now(),
			PreviousThresholds: oldThresholds,
			NewThresholds:      nr.thresholds,
			Reason:             fmt.Sprintf("Network health changed from %.2f to %.2f", nr.networkHealth, health),
			AffectedNodeCount:  len(nr.scores),
		}
		nr.adaptationHistory = append(nr.adaptationHistory, adjustment)

		// Limit history size
		if len(nr.adaptationHistory) > 100 {
			nr.adaptationHistory = nr.adaptationHistory[1:]
		}
	}

	nr.networkHealth = health
}

// updateTrendDirection calculates the trend in node reputation over time
func (nr *NodeReputation) updateTrendDirection(score *ReputationScore) {
	// Simple linear regression to find trend
	n := len(score.HistoricalScores)
	if n < 2 {
		score.TrendDirection = 0
		return
	}

	// Calculate time-based x values (seconds since first timestamp)
	xValues := make([]float64, n)
	baseTime := score.ScoringTimes[0]
	for i, timestamp := range score.ScoringTimes {
		xValues[i] = timestamp.Sub(baseTime).Seconds()
	}

	// Calculate means
	var sumX, sumY, sumXY, sumXX float64
	for i, x := range xValues {
		y := score.HistoricalScores[i]
		sumX += x
		sumY += y
		sumXY += x * y
		sumXX += x * x
	}

	meanX := sumX / float64(n)
	meanY := sumY / float64(n)

	// Calculate slope
	denominator := sumXX - float64(n)*meanX*meanX
	if denominator == 0 {
		score.TrendDirection = 0
		return
	}

	slope := (sumXY - float64(n)*meanX*meanY) / denominator
	score.TrendDirection = slope

	// Calculate standard deviation
	var sumSquaredDiff float64
	for _, y := range score.HistoricalScores {
		diff := y - meanY
		sumSquaredDiff += diff * diff
	}
	score.StandardDeviation = math.Sqrt(sumSquaredDiff / float64(n))

	// Calculate consistency score (inverse of coefficient of variation, capped)
	if meanY > 0 {
		cv := score.StandardDeviation / meanY // coefficient of variation
		score.ConsistencyScore = math.Max(0, 1.0-cv)
	} else {
		score.ConsistencyScore = 0
	}
}

// updatePerformanceModel updates the node's performance predictive model
func (nr *NodeReputation) updatePerformanceModel(nodeID string, currentScore float64) {
	model, exists := nr.performanceModels[nodeID]
	if !exists {
		model = &PerformanceModel{
			NodeID:           nodeID,
			ExpectedScore:    currentScore,
			Variance:         0.01,
			ConfidenceLevel:  0.5,
			LastUpdated:      time.Now(),
			PredictionErrors: make([]float64, 0),
		}
		nr.performanceModels[nodeID] = model
	} else {
		// Calculate prediction error
		error := currentScore - model.ExpectedScore
		model.PredictionErrors = append(model.PredictionErrors, error)

		// Limit history
		if len(model.PredictionErrors) > 10 {
			model.PredictionErrors = model.PredictionErrors[1:]
		}

		// Update expected score with exponential moving average
		model.ExpectedScore = 0.8*model.ExpectedScore + 0.2*currentScore

		// Update variance
		if len(model.PredictionErrors) >= 3 {
			var sumSquaredErrors float64
			for _, err := range model.PredictionErrors {
				sumSquaredErrors += err * err
			}
			model.Variance = sumSquaredErrors / float64(len(model.PredictionErrors))
		}

		// Update confidence based on consistent predictions
		if model.Variance < 0.01 {
			model.ConfidenceLevel = math.Min(1.0, model.ConfidenceLevel+0.05)
		} else if model.Variance > 0.05 {
			model.ConfidenceLevel = math.Max(0.1, model.ConfidenceLevel-0.05)
		}

		model.LastUpdated = time.Now()
	}
}

// detectAnomaly checks if the current score is anomalous compared to expected performance
func (nr *NodeReputation) detectAnomaly(nodeID string, currentScore float64) bool {
	model, exists := nr.performanceModels[nodeID]
	if !exists || model.ConfidenceLevel < 0.6 {
		return false // Not enough data for reliable anomaly detection
	}

	// Calculate z-score
	if model.Variance == 0 {
		return false
	}

	stdDev := math.Sqrt(model.Variance)
	zScore := math.Abs(currentScore-model.ExpectedScore) / stdDev

	// Z-score > 2 means the value is outside the 95% confidence interval
	// Scale threshold by confidence level
	anomalyThreshold := 2.0 * (2.0 - model.ConfidenceLevel)
	return zScore > anomalyThreshold
}

// calculateNodeScore calculates a reputation score based on node history
func (nr *NodeReputation) calculateNodeScore(score *ReputationScore) float64 {
	if score.TotalInteractions == 0 {
		return 0.5 // Default score for no interactions
	}

	// Base score from successful vs. failed actions
	baseScore := float64(score.SuccessfulActions) / float64(score.TotalInteractions)

	// Apply penalty for suspicious patterns
	suspiciousPenalty := math.Min(float64(score.SuspiciousPatterns)/float64(score.TotalInteractions)*2.0, 0.8)

	// Apply penalty for anomalies
	anomalyPenalty := math.Min(float64(score.AnomalyCount)*0.05, 0.3)

	// Boost for consistency if available
	consistencyBoost := 0.0
	if score.ConsistencyScore > 0 {
		consistencyBoost = score.ConsistencyScore * 0.1 // Max 10% boost
	}

	// Final score calculation
	finalScore := baseScore*(1.0-suspiciousPenalty)*(1.0-anomalyPenalty) + consistencyBoost

	// Ensure score is in range [0,1]
	return math.Max(0.0, math.Min(1.0, finalScore))
}

// GetNodeTrend gets the reputation trend for a node
func (nr *NodeReputation) GetNodeTrend(nodeID string) (trend float64, confidence float64) {
	nr.mutex.RLock()
	defer nr.mutex.RUnlock()

	score, exists := nr.scores[nodeID]
	if !exists {
		return 0, 0 // No data
	}

	model, hasModel := nr.performanceModels[nodeID]
	if !hasModel {
		return score.TrendDirection, 0.5 // Basic confidence
	}

	return score.TrendDirection, model.ConfidenceLevel
}

// GetReputationHistory gets historical reputation data for a node
func (nr *NodeReputation) GetReputationHistory(nodeID string) (scores []float64, timestamps []time.Time) {
	nr.mutex.RLock()
	defer nr.mutex.RUnlock()

	score, exists := nr.scores[nodeID]
	if !exists {
		return nil, nil
	}

	// Return copies to avoid modification
	scoresCopy := make([]float64, len(score.HistoricalScores))
	timesCopy := make([]time.Time, len(score.ScoringTimes))

	copy(scoresCopy, score.HistoricalScores)
	copy(timesCopy, score.ScoringTimes)

	return scoresCopy, timesCopy
}

// PredictFutureReputation predicts a node's reputation in the future
func (nr *NodeReputation) PredictFutureReputation(nodeID string, duration time.Duration) (predictedScore float64, confidence float64) {
	nr.mutex.RLock()
	defer nr.mutex.RUnlock()

	score, exists := nr.scores[nodeID]
	if !exists {
		return 0.5, 0.1 // Default with low confidence
	}

	model, hasModel := nr.performanceModels[nodeID]
	if !hasModel {
		return nr.GetReputationScore(nodeID), 0.3 // Current score with low confidence
	}

	// Use trend to predict future score
	predictedScore = model.ExpectedScore + (score.TrendDirection * duration.Seconds() / 86400.0) // Trend per day

	// Ensure prediction is in valid range
	predictedScore = math.Max(0.0, math.Min(1.0, predictedScore))

	// Lower confidence for longer predictions
	daysInFuture := duration.Hours() / 24.0
	confDecay := math.Min(daysInFuture*0.1, 0.9) // Lose up to 90% confidence
	adjustedConfidence := model.ConfidenceLevel * (1.0 - confDecay)

	return predictedScore, adjustedConfidence
}
