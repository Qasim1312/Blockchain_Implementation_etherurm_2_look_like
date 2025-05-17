// pkg/cap/orchestrator.go
package cap

import (
	"fmt"
	"log"
	"math"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/qasim/blockchain_assignment_3/pkg/utils"
)

// ConsistencyLevel represents different levels of consistency
type ConsistencyLevel int

const (
	StrongConsistency ConsistencyLevel = iota
	CausalConsistency
	EventualConsistency
)

// ConsistencyDimension represents a dimension of consistency measurement
type ConsistencyDimension struct {
	Name      string
	Weight    float64
	RawValue  float64
	Threshold float64
}

// NetworkStats tracks network performance metrics
type NetworkStats struct {
	LatencyMs           int64
	PartitionEvents     int
	PacketLossRate      float64
	LastPartitionEnd    time.Time
	JitterMs            int64
	BandwidthMbps       float64
	HistoricalLatencies []int64
	LatencyTrend        float64 // Positive means increasing latency
	NodeCount           int
	DisconnectedNodes   []string
	RegionalStats       map[string]*RegionalNetworkStats
}

// RegionalNetworkStats tracks network metrics for a specific region
type RegionalNetworkStats struct {
	Region              string
	LatencyMs           int64
	PacketLossRate      float64
	ConnectivityHistory []bool // True = connected, False = disconnected
	LastUpdated         time.Time
}

// MultiDimensionalConsistencyOrchestrator dynamically adjusts consistency levels
type MultiDimensionalConsistencyOrchestrator struct {
	currentLevel       ConsistencyLevel
	networkStats       NetworkStats
	dimensions         map[string]*ConsistencyDimension
	partitionRisk      float64
	mutex              sync.RWMutex
	adjustmentTicker   *time.Ticker
	networkProbe       *NetworkProbe
	adaptiveTimeouts   map[string]time.Duration
	consistencyPolicy  ConsistencyPolicy
	adaptationHistory  []ConsistencyAdaptation
	nodeHealthCheckers map[string]*NodeHealthChecker
}

// ConsistencyPolicy defines rules for consistency adaptation
type ConsistencyPolicy struct {
	PrioritizeAvailability bool
	MinAcceptableLatency   int64
	MaxAcceptableLatency   int64
	AutoAdjust             bool
	RegionalOverrides      map[string]ConsistencyLevel
}

// ConsistencyAdaptation tracks a history of consistency adaptations
type ConsistencyAdaptation struct {
	Timestamp     time.Time
	PreviousLevel ConsistencyLevel
	NewLevel      ConsistencyLevel
	Reason        string
	NetworkState  NetworkStats
}

// NetworkProbe performs active network measurements
type NetworkProbe struct {
	targets         []string
	probeInterval   time.Duration
	lastProbeResult map[string]ProbeResult
	mutex           sync.RWMutex
	isRunning       bool
}

// ProbeResult contains the results of a network probe
type ProbeResult struct {
	Target     string
	LatencyMs  int64
	PacketLoss float64
	Timestamp  time.Time
	Successful bool
}

// NodeHealthChecker monitors the health of a specific node
type NodeHealthChecker struct {
	NodeID        string
	LastSeen      time.Time
	Status        string
	FailureCount  int
	ResponseTimes []time.Duration
	mutex         sync.RWMutex
}

// NewConsistencyOrchestrator creates a new consistency orchestrator
func NewConsistencyOrchestrator() *MultiDimensionalConsistencyOrchestrator {
	orchestrator := &MultiDimensionalConsistencyOrchestrator{
		currentLevel:       CausalConsistency, // Default to middle ground
		networkStats:       NetworkStats{RegionalStats: make(map[string]*RegionalNetworkStats)},
		partitionRisk:      0.1, // Initial partition risk assessment
		dimensions:         make(map[string]*ConsistencyDimension),
		adaptiveTimeouts:   make(map[string]time.Duration),
		adaptationHistory:  make([]ConsistencyAdaptation, 0),
		nodeHealthCheckers: make(map[string]*NodeHealthChecker),
		consistencyPolicy: ConsistencyPolicy{
			PrioritizeAvailability: true,
			MinAcceptableLatency:   50,  // 50ms
			MaxAcceptableLatency:   500, // 500ms
			AutoAdjust:             true,
			RegionalOverrides:      make(map[string]ConsistencyLevel),
		},
	}

	// Initialize consistency dimensions
	orchestrator.dimensions["latency"] = &ConsistencyDimension{
		Name:      "latency",
		Weight:    0.4,
		RawValue:  0.0,
		Threshold: 0.6,
	}

	orchestrator.dimensions["packetLoss"] = &ConsistencyDimension{
		Name:      "packetLoss",
		Weight:    0.3,
		RawValue:  0.0,
		Threshold: 0.1,
	}

	orchestrator.dimensions["partitionHistory"] = &ConsistencyDimension{
		Name:      "partitionHistory",
		Weight:    0.2,
		RawValue:  0.0,
		Threshold: 0.5,
	}

	orchestrator.dimensions["nodeCount"] = &ConsistencyDimension{
		Name:      "nodeCount",
		Weight:    0.1,
		RawValue:  1.0, // Start with 1 node
		Threshold: 5.0, // Threshold for considering cluster size
	}

	// Initialize network probe
	orchestrator.networkProbe = &NetworkProbe{
		targets:         []string{},
		probeInterval:   10 * time.Second,
		lastProbeResult: make(map[string]ProbeResult),
		isRunning:       false,
	}

	// Start periodic adjustment
	orchestrator.adjustmentTicker = time.NewTicker(10 * time.Second)
	go orchestrator.periodicAdjustment()

	// Start network probing
	go orchestrator.startNetworkProbing()

	return orchestrator
}

// GetConsistencyLevel returns the current consistency level
func (o *MultiDimensionalConsistencyOrchestrator) GetConsistencyLevel() ConsistencyLevel {
	o.mutex.RLock()
	defer o.mutex.RUnlock()
	return o.currentLevel
}

// UpdateNetworkStats updates network statistics
func (o *MultiDimensionalConsistencyOrchestrator) UpdateNetworkStats(latency int64, packetLoss float64) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	// Store historical latency for trend analysis
	o.networkStats.HistoricalLatencies = append(o.networkStats.HistoricalLatencies, latency)
	if len(o.networkStats.HistoricalLatencies) > 10 {
		o.networkStats.HistoricalLatencies = o.networkStats.HistoricalLatencies[1:]
	}

	// Calculate latency trend
	if len(o.networkStats.HistoricalLatencies) >= 5 {
		o.calculateLatencyTrend()
	}

	o.networkStats.LatencyMs = latency
	o.networkStats.PacketLossRate = packetLoss

	// Update dimension values
	o.dimensions["latency"].RawValue = float64(latency) / float64(o.consistencyPolicy.MaxAcceptableLatency)
	o.dimensions["packetLoss"].RawValue = packetLoss

	// Adjust partition risk based on network conditions
	o.calculatePartitionRisk()

	// Adjust consistency if automatic adjustment is enabled
	if o.consistencyPolicy.AutoAdjust {
		o.adjustConsistencyLevel()
	}
}

// calculateLatencyTrend calculates the trend in latency
func (o *MultiDimensionalConsistencyOrchestrator) calculateLatencyTrend() {
	n := len(o.networkStats.HistoricalLatencies)
	if n < 2 {
		o.networkStats.LatencyTrend = 0
		return
	}

	// Calculate linear regression slope
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0

	for i, latency := range o.networkStats.HistoricalLatencies {
		x := float64(i)
		y := float64(latency)
		sumX += x
		sumY += y
		sumXY += x * y
		sumXX += x * x
	}

	// Calculate slope
	denominator := float64(n)*sumXX - sumX*sumX
	if denominator != 0 {
		o.networkStats.LatencyTrend = (float64(n)*sumXY - sumX*sumY) / denominator
	} else {
		o.networkStats.LatencyTrend = 0
	}
}

// NotifyPartitionEvent signals that a network partition was detected
func (o *MultiDimensionalConsistencyOrchestrator) NotifyPartitionEvent(recovered bool, affectedRegion string) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	if recovered {
		o.networkStats.LastPartitionEnd = time.Now()

		// Update regional stats if applicable
		if affectedRegion != "" && o.networkStats.RegionalStats[affectedRegion] != nil {
			o.networkStats.RegionalStats[affectedRegion].ConnectivityHistory = append(
				o.networkStats.RegionalStats[affectedRegion].ConnectivityHistory, true)
		}
	} else {
		o.networkStats.PartitionEvents++
		o.partitionRisk = 0.8 // High risk during known partition

		// Update regional stats if applicable
		if affectedRegion != "" {
			if _, exists := o.networkStats.RegionalStats[affectedRegion]; !exists {
				o.networkStats.RegionalStats[affectedRegion] = &RegionalNetworkStats{
					Region:              affectedRegion,
					ConnectivityHistory: make([]bool, 0),
					LastUpdated:         time.Now(),
				}
			}

			o.networkStats.RegionalStats[affectedRegion].ConnectivityHistory = append(
				o.networkStats.RegionalStats[affectedRegion].ConnectivityHistory, false)
		}

		// Update partition history dimension
		historyValue := float64(o.networkStats.PartitionEvents) / 10.0
		if historyValue > 1.0 {
			historyValue = 1.0
		}
		o.dimensions["partitionHistory"].RawValue = historyValue
	}

	// Force immediate consistency adjustment
	o.adjustConsistencyLevel()

	// Log adaptation
	o.recordAdaptation("Partition event: " + affectedRegion)
}

// UpdateRegionalStats updates network statistics for a specific region
func (o *MultiDimensionalConsistencyOrchestrator) UpdateRegionalStats(region string, latency int64, packetLoss float64) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	// Initialize regional stats if needed
	if _, exists := o.networkStats.RegionalStats[region]; !exists {
		o.networkStats.RegionalStats[region] = &RegionalNetworkStats{
			Region:              region,
			ConnectivityHistory: make([]bool, 0),
			LastUpdated:         time.Now(),
		}
	}

	// Update stats
	regionalStats := o.networkStats.RegionalStats[region]
	regionalStats.LatencyMs = latency
	regionalStats.PacketLossRate = packetLoss
	regionalStats.LastUpdated = time.Now()

	// Consider regional override if applicable
	if level, hasOverride := o.consistencyPolicy.RegionalOverrides[region]; hasOverride {
		log.Printf("[CAP] Applied regional consistency override for %s: %v", region, level)
	}
}

// RegisterNode registers a node for health checking
func (o *MultiDimensionalConsistencyOrchestrator) RegisterNode(nodeID string) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	if _, exists := o.nodeHealthCheckers[nodeID]; !exists {
		o.nodeHealthCheckers[nodeID] = &NodeHealthChecker{
			NodeID:        nodeID,
			LastSeen:      time.Now(),
			Status:        "healthy",
			FailureCount:  0,
			ResponseTimes: make([]time.Duration, 0),
			mutex:         sync.RWMutex{},
		}
	}

	// Update node count dimension
	o.networkStats.NodeCount = len(o.nodeHealthCheckers)
	o.dimensions["nodeCount"].RawValue = float64(o.networkStats.NodeCount)
}

// UpdateNodeHealth updates the health status of a node
func (o *MultiDimensionalConsistencyOrchestrator) UpdateNodeHealth(nodeID string, healthy bool, responseTime time.Duration) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	checker, exists := o.nodeHealthCheckers[nodeID]
	if !exists {
		return
	}

	checker.mutex.Lock()
	defer checker.mutex.Unlock()

	if healthy {
		checker.LastSeen = time.Now()
		checker.Status = "healthy"
		checker.FailureCount = 0
	} else {
		checker.FailureCount++
		if checker.FailureCount > 3 {
			checker.Status = "unhealthy"

			// Add to disconnected nodes list if not already present
			found := false
			for _, id := range o.networkStats.DisconnectedNodes {
				if id == nodeID {
					found = true
					break
				}
			}

			if !found {
				o.networkStats.DisconnectedNodes = append(o.networkStats.DisconnectedNodes, nodeID)

				// If we've lost more than 25% of nodes, consider this a partition
				disconnectedRatio := float64(len(o.networkStats.DisconnectedNodes)) / float64(o.networkStats.NodeCount)
				if disconnectedRatio > 0.25 {
					o.NotifyPartitionEvent(false, "")
				}
			}
		}
	}

	// Store response time for timeout calculations
	checker.ResponseTimes = append(checker.ResponseTimes, responseTime)
	if len(checker.ResponseTimes) > 10 {
		checker.ResponseTimes = checker.ResponseTimes[1:]
	}

	// Update adaptive timeout for this node
	o.calculateNodeTimeout(nodeID)
}

// calculateNodeTimeout calculates an appropriate timeout for a specific node
func (o *MultiDimensionalConsistencyOrchestrator) calculateNodeTimeout(nodeID string) {
	checker := o.nodeHealthCheckers[nodeID]

	if len(checker.ResponseTimes) == 0 {
		o.adaptiveTimeouts[nodeID] = 500 * time.Millisecond
		return
	}

	// Calculate average response time
	var totalTime time.Duration
	for _, t := range checker.ResponseTimes {
		totalTime += t
	}
	avgTime := totalTime / time.Duration(len(checker.ResponseTimes))

	// Calculate standard deviation
	var variance float64
	for _, t := range checker.ResponseTimes {
		diff := float64(t - avgTime)
		variance += diff * diff
	}
	variance /= float64(len(checker.ResponseTimes))
	stdDev := time.Duration(math.Sqrt(variance))

	// Set timeout as average + 2 standard deviations (95% confidence)
	// with bounds to prevent extremes
	timeout := avgTime + 2*stdDev

	// Apply consistency level factor
	switch o.currentLevel {
	case StrongConsistency:
		timeout = timeout * 3 // Longer timeouts for strong consistency
	case CausalConsistency:
		timeout = timeout * 2 // Medium timeouts
	case EventualConsistency:
		// No multiplier for eventual consistency
	}

	// Enforce minimum and maximum bounds
	if timeout < 100*time.Millisecond {
		timeout = 100 * time.Millisecond
	} else if timeout > 5*time.Second {
		timeout = 5 * time.Second
	}

	o.adaptiveTimeouts[nodeID] = timeout
}

// calculatePartitionRisk estimates the probability of network partitions
func (o *MultiDimensionalConsistencyOrchestrator) calculatePartitionRisk() {
	// Calculate a weighted sum of all dimensions
	weightedSum := 0.0
	totalWeight := 0.0

	for _, dim := range o.dimensions {
		weightedSum += dim.RawValue * dim.Weight
		totalWeight += dim.Weight
	}

	// Normalize to get final risk
	if totalWeight > 0 {
		o.partitionRisk = weightedSum / totalWeight
	} else {
		o.partitionRisk = 0.5 // Default if no weights
	}

	// Add extra risk based on latency trend
	if o.networkStats.LatencyTrend > 0 {
		// Positive trend means increasing latency - higher risk
		trendRisk := math.Min(o.networkStats.LatencyTrend/100.0, 0.2)
		o.partitionRisk += trendRisk
	}

	// Ensure risk is in valid range
	if o.partitionRisk < 0.0 {
		o.partitionRisk = 0.0
	} else if o.partitionRisk > 1.0 {
		o.partitionRisk = 1.0
	}

	// Add logging to calculatePartitionRisk method
	utils.LogToFile("cap_partitions", fmt.Sprintf("Current network partition probability: %.4f (latency: %dms, trend: %.3f, timeout rate: %.2f%%)",
		o.partitionRisk, o.networkStats.LatencyMs, o.networkStats.LatencyTrend, o.networkStats.PacketLossRate*100))
}

// adjustConsistencyLevel dynamically adjusts the consistency level
func (o *MultiDimensionalConsistencyOrchestrator) adjustConsistencyLevel() {
	previousLevel := o.currentLevel

	// Decision thresholds
	const (
		lowRisk    = 0.2
		mediumRisk = 0.6
	)

	// Use policy rules to guide decision
	if o.consistencyPolicy.PrioritizeAvailability {
		// Bias toward availability
		if o.partitionRisk > mediumRisk {
			o.currentLevel = EventualConsistency
		} else if o.partitionRisk > lowRisk {
			o.currentLevel = CausalConsistency
		} else {
			o.currentLevel = StrongConsistency
		}
	} else {
		// Bias toward consistency
		if o.partitionRisk < lowRisk {
			o.currentLevel = StrongConsistency
		} else if o.partitionRisk < mediumRisk {
			o.currentLevel = CausalConsistency
		} else {
			o.currentLevel = EventualConsistency
		}
	}

	// Record adaptation if level changed
	if previousLevel != o.currentLevel {
		reason := "Partition risk changed to " + fmt.Sprintf("%.2f", o.partitionRisk)
		o.recordAdaptation(reason)
	}

	// Add logging to adjustConsistencyLevel method
	utils.LogToFile("cap_consistency", fmt.Sprintf("Adjusted consistency level from %.2f to %.2f based on partition risk: %.3f",
		float64(previousLevel), float64(o.currentLevel), o.partitionRisk))
}

// recordAdaptation records a consistency adaptation event
func (o *MultiDimensionalConsistencyOrchestrator) recordAdaptation(reason string) {
	// Deep copy network stats
	statsCopy := NetworkStats{
		LatencyMs:       o.networkStats.LatencyMs,
		PartitionEvents: o.networkStats.PartitionEvents,
		PacketLossRate:  o.networkStats.PacketLossRate,
		JitterMs:        o.networkStats.JitterMs,
		BandwidthMbps:   o.networkStats.BandwidthMbps,
		LatencyTrend:    o.networkStats.LatencyTrend,
		NodeCount:       o.networkStats.NodeCount,
	}

	adaptation := ConsistencyAdaptation{
		Timestamp:     time.Now(),
		PreviousLevel: o.currentLevel,
		NewLevel:      o.currentLevel,
		Reason:        reason,
		NetworkState:  statsCopy,
	}

	o.adaptationHistory = append(o.adaptationHistory, adaptation)

	// Keep history bounded
	if len(o.adaptationHistory) > 100 {
		o.adaptationHistory = o.adaptationHistory[len(o.adaptationHistory)-100:]
	}

	log.Printf("[CAP] Consistency adapted to %v: %s", o.currentLevel, reason)
}

// periodicAdjustment runs consistency adjustments periodically
func (o *MultiDimensionalConsistencyOrchestrator) periodicAdjustment() {
	for range o.adjustmentTicker.C {
		o.mutex.Lock()
		o.calculatePartitionRisk()
		if o.consistencyPolicy.AutoAdjust {
			o.adjustConsistencyLevel()
		}
		o.mutex.Unlock()
	}
}

// startNetworkProbing begins active network probing
func (o *MultiDimensionalConsistencyOrchestrator) startNetworkProbing() {
	if o.networkProbe.isRunning {
		return
	}

	o.networkProbe.isRunning = true
	ticker := time.NewTicker(o.networkProbe.probeInterval)

	for range ticker.C {
		o.probeTargets()
	}
}

// probeTargets probes all registered targets for network statistics
func (o *MultiDimensionalConsistencyOrchestrator) probeTargets() {
	o.networkProbe.mutex.RLock()
	targets := o.networkProbe.targets
	o.networkProbe.mutex.RUnlock()

	for _, target := range targets {
		go o.probeTarget(target)
	}
}

// probeTarget measures network quality to a specific target
func (o *MultiDimensionalConsistencyOrchestrator) probeTarget(target string) {
	start := time.Now()

	// Attempt to connect to the target
	conn, err := net.DialTimeout("tcp", target, 2*time.Second)

	result := ProbeResult{
		Target:     target,
		Timestamp:  time.Now(),
		Successful: err == nil,
	}

	if err == nil {
		// Calculate latency
		result.LatencyMs = time.Since(start).Milliseconds()
		conn.Close()
	} else {
		// Connection failed
		result.PacketLoss = 1.0
	}

	// Store result
	o.networkProbe.mutex.Lock()
	o.networkProbe.lastProbeResult[target] = result
	o.networkProbe.mutex.Unlock()

	// Update network stats if successful
	if result.Successful {
		o.mutex.Lock()

		// Use exponential moving average for latency
		o.networkStats.LatencyMs = (o.networkStats.LatencyMs*7 + result.LatencyMs) / 8

		// Extract region from target if possible
		region := extractRegionFromTarget(target)
		if region != "" {
			o.UpdateRegionalStats(region, result.LatencyMs, result.PacketLoss)
		}

		o.mutex.Unlock()
	}
}

// AddProbeTarget adds a target for network probing
func (o *MultiDimensionalConsistencyOrchestrator) AddProbeTarget(target string) {
	o.networkProbe.mutex.Lock()
	defer o.networkProbe.mutex.Unlock()

	// Check if target already exists
	for _, existingTarget := range o.networkProbe.targets {
		if existingTarget == target {
			return
		}
	}

	o.networkProbe.targets = append(o.networkProbe.targets, target)
}

// GetProbeResults returns the results of the most recent probes
func (o *MultiDimensionalConsistencyOrchestrator) GetProbeResults() map[string]ProbeResult {
	o.networkProbe.mutex.RLock()
	defer o.networkProbe.mutex.RUnlock()

	// Copy results to avoid concurrent modification
	results := make(map[string]ProbeResult)
	for k, v := range o.networkProbe.lastProbeResult {
		results[k] = v
	}

	return results
}

// GetNodeTimeout gets the adaptive timeout for a specific node
func (o *MultiDimensionalConsistencyOrchestrator) GetNodeTimeout(nodeID string) time.Duration {
	o.mutex.RLock()
	defer o.mutex.RUnlock()

	timeout, exists := o.adaptiveTimeouts[nodeID]
	if !exists {
		// Return a default timeout if not specifically calculated
		return o.GetTimeout()
	}

	return timeout
}

// GetTimeout calculates appropriate timeout based on network conditions
func (o *MultiDimensionalConsistencyOrchestrator) GetTimeout() time.Duration {
	o.mutex.RLock()
	defer o.mutex.RUnlock()

	// Base timeout
	baseTimeout := 500 * time.Millisecond

	// Adjust based on consistency level
	switch o.currentLevel {
	case StrongConsistency:
		// Longer timeouts for strong consistency
		baseTimeout *= 3
	case CausalConsistency:
		// Medium timeouts
		baseTimeout *= 2
	case EventualConsistency:
		// Shorter timeouts for eventual consistency
	}

	// Adjust for network conditions
	networkFactor := 1.0 + float64(o.networkStats.LatencyMs)/100.0
	if networkFactor > 5.0 {
		networkFactor = 5.0 // Cap the multiplier
	}

	finalTimeout := time.Duration(float64(baseTimeout) * networkFactor)
	return finalTimeout
}

// GetPartitionRisk returns the current partition risk assessment
func (o *MultiDimensionalConsistencyOrchestrator) GetPartitionRisk() float64 {
	o.mutex.RLock()
	defer o.mutex.RUnlock()
	return o.partitionRisk
}

// SetConsistencyPolicy sets the consistency policy
func (o *MultiDimensionalConsistencyOrchestrator) SetConsistencyPolicy(policy ConsistencyPolicy) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	o.consistencyPolicy = policy

	// Force recalculation with new policy
	o.calculatePartitionRisk()
	if policy.AutoAdjust {
		o.adjustConsistencyLevel()
	}
}

// GetAdaptationHistory returns the consistency adaptation history
func (o *MultiDimensionalConsistencyOrchestrator) GetAdaptationHistory() []ConsistencyAdaptation {
	o.mutex.RLock()
	defer o.mutex.RUnlock()

	// Copy history to avoid concurrent modification
	history := make([]ConsistencyAdaptation, len(o.adaptationHistory))
	copy(history, o.adaptationHistory)

	return history
}

// extractRegionFromTarget extracts region info from a target address
func extractRegionFromTarget(target string) string {
	// This is a simplified example
	// In a real system, this would use DNS or metadata

	// Example: "us-west.example.com:8080" -> "us-west"
	parts := strings.Split(target, ".")
	if len(parts) > 0 {
		regionParts := strings.Split(parts[0], "-")
		if len(regionParts) >= 2 {
			return parts[0]
		}
	}

	return ""
}
