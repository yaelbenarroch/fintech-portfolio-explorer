
import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChartContainer } from "@/components/ui/chart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import * as Recharts from "recharts";
import { Brain, ArrowLeft, LineChart, Briefcase, AlertCircle } from "lucide-react";

const MLDashboard = () => {
  // State for selected asset
  const [selectedAsset, setSelectedAsset] = useState("Bitcoin");
  const [predictionDays, setPredictionDays] = useState(7);
  const [threshold, setThreshold] = useState(3.0);

  // Sample data for portfolio
  const assets = [
    { name: "Bitcoin", ticker: "BTC-USD" },
    { name: "Ethereum", ticker: "ETH-USD" },
    { name: "Solana", ticker: "SOL-USD" },
    { name: "Cardano", ticker: "ADA-USD" },
    { name: "Polkadot", ticker: "DOT-USD" }
  ];

  // Sample prediction data
  const generatePredictionData = (days: number) => {
    const today = new Date();
    const data = [];

    // Historical data (last 30 days)
    for (let i = 30; i > 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      const price = Math.random() * 10000 + 40000; // Random price between 40k-50k for BTC
      data.push({
        date: date.toISOString().split('T')[0],
        price: price,
        type: 'Historical'
      });
    }

    // Future predictions
    let lastPrice = data[data.length - 1].price;
    for (let i = 1; i <= days; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      
      // Simulate a prediction with some randomness but follow the trend
      const randomFactor = Math.random() * 0.05 - 0.025; // -2.5% to +2.5%
      lastPrice = lastPrice * (1 + randomFactor);
      
      data.push({
        date: date.toISOString().split('T')[0],
        price: lastPrice,
        type: 'Predicted'
      });
    }

    return data;
  };

  // Generate sample data for features importance
  const featureImportanceData = [
    { feature: "Price_Lag1", importance: 0.35 },
    { feature: "Volume", importance: 0.25 },
    { feature: "MA5", importance: 0.15 },
    { feature: "Volatility", importance: 0.12 },
    { feature: "RSI", importance: 0.08 },
    { feature: "MACD", importance: 0.05 }
  ];

  // Generate efficient frontier data
  const generateEfficientFrontier = () => {
    const data = [];
    for (let i = 0; i < 100; i++) {
      // Generate random risk-return points
      const volatility = Math.random() * 0.3 + 0.1; // 10-40% volatility
      const expectedReturn = volatility * (1 + Math.random() * 0.5); // Return increases with volatility
      const sharpe = expectedReturn / volatility;
      
      data.push({
        volatility,
        return: expectedReturn,
        sharpe
      });
    }
    
    // Add current and optimized portfolio points
    data.push({
      volatility: 0.25,
      return: 0.2,
      sharpe: 0.8,
      portfolio: 'Current'
    });
    
    data.push({
      volatility: 0.22,
      return: 0.25,
      sharpe: 1.14,
      portfolio: 'Optimized'
    });
    
    return data;
  };

  // Generate anomaly detection data
  const generateAnomalyData = (threshold: number) => {
    const data = [];
    const today = new Date();
    let lastPrice = 45000;
    
    // Generate 90 days of price data
    for (let i = 90; i > 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      
      // Normal price movement with random walk
      const normalMovement = (Math.random() - 0.5) * 0.02; // -1% to +1% daily change
      
      // Add occasional anomalies (price spikes or drops)
      let anomalyFactor = 0;
      let isAnomaly = false;
      
      // Create an anomaly roughly once every 15 days
      if (i % 15 === 0) {
        anomalyFactor = (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 0.08 + 0.05); // 5-13% spike or drop
        isAnomaly = true;
      }
      
      lastPrice = lastPrice * (1 + normalMovement + anomalyFactor);
      
      // Calculate z-score (simplified)
      const zScore = isAnomaly ? (Math.random() * 2 + threshold) * (anomalyFactor > 0 ? 1 : -1) : (Math.random() * 1.5);
      
      data.push({
        date: date.toISOString().split('T')[0],
        price: lastPrice,
        zScore: zScore,
        isAnomaly: isAnomaly
      });
    }
    
    return data;
  };

  // Get prediction data based on selected days
  const predictionData = generatePredictionData(predictionDays);
  const efficientFrontierData = generateEfficientFrontier();
  const anomalyData = generateAnomalyData(threshold);

  return (
    <div className="container mx-auto p-4">
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-purple-700 flex items-center">
            <Brain className="mr-2 h-6 w-6" />
            ML Analytics Dashboard
          </h1>
          <p className="text-gray-600">
            Advanced machine learning models for cryptocurrency analysis
          </p>
        </div>
        <Link to="/">
          <Button variant="outline" className="flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Overview
          </Button>
        </Link>
      </header>

      <Tabs defaultValue="prediction" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="prediction" className="flex items-center">
            <LineChart className="mr-2 h-4 w-4" /> Price Prediction
          </TabsTrigger>
          <TabsTrigger value="optimization" className="flex items-center">
            <Briefcase className="mr-2 h-4 w-4" /> Portfolio Optimization
          </TabsTrigger>
          <TabsTrigger value="anomaly" className="flex items-center">
            <AlertCircle className="mr-2 h-4 w-4" /> Anomaly Detection
          </TabsTrigger>
        </TabsList>

        {/* Price Prediction Tab */}
        <TabsContent value="prediction" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Machine Learning Price Prediction</CardTitle>
              <CardDescription>
                Random Forest model trained on historical prices, technical indicators, and market sentiment
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="w-full md:w-1/4">
                  <Select value={selectedAsset} onValueChange={setSelectedAsset}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select Asset" />
                    </SelectTrigger>
                    <SelectContent>
                      {assets.map((asset) => (
                        <SelectItem key={asset.ticker} value={asset.name}>
                          {asset.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="w-full md:w-3/4">
                  <div className="flex flex-col space-y-2">
                    <label className="text-sm font-medium">
                      Prediction Days: {predictionDays}
                    </label>
                    <Slider
                      min={1}
                      max={30}
                      step={1}
                      value={[predictionDays]}
                      onValueChange={(value) => setPredictionDays(value[0])}
                    />
                  </div>
                </div>
              </div>

              <div className="h-[400px]">
                <ChartContainer config={{}} className="h-full">
                  <Recharts.ComposedChart data={predictionData}>
                    <Recharts.CartesianGrid strokeDasharray="3 3" />
                    <Recharts.XAxis dataKey="date" />
                    <Recharts.YAxis />
                    <Recharts.Tooltip />
                    <Recharts.Legend />
                    <Recharts.Line
                      type="monotone"
                      dataKey="price"
                      stroke="#8884d8"
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                      name={`${selectedAsset} Price`}
                    />
                    <Recharts.Area
                      type="monotone"
                      dataKey={(data) => (data.type === 'Predicted' ? data.price : undefined)}
                      fill="#8884d8"
                      fillOpacity={0.1}
                      stroke="none"
                      name="Prediction Range"
                    />
                    <Recharts.ReferenceArea
                      x1={predictionData.findIndex(d => d.type === 'Predicted') > 0 ? 
                        predictionData[predictionData.findIndex(d => d.type === 'Predicted') - 1].date : 
                        predictionData[0].date}
                      x2={predictionData[predictionData.length - 1].date}
                      fill="#8884d8"
                      fillOpacity={0.1}
                      stroke="none"
                    />
                  </Recharts.ComposedChart>
                </ChartContainer>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Feature Importance</CardTitle>
                    <CardDescription>
                      Key factors influencing the prediction model
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[250px]">
                      <ChartContainer config={{}} className="h-full">
                        <Recharts.BarChart data={featureImportanceData}>
                          <Recharts.CartesianGrid strokeDasharray="3 3" />
                          <Recharts.XAxis dataKey="feature" />
                          <Recharts.YAxis />
                          <Recharts.Tooltip />
                          <Recharts.Bar dataKey="importance" fill="#8884d8" name="Importance" />
                        </Recharts.BarChart>
                      </ChartContainer>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Model Performance</CardTitle>
                    <CardDescription>
                      Evaluation metrics for the prediction model
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Root Mean Square Error</span>
                        <span className="text-sm font-bold">$345.67</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Mean Absolute Percentage Error</span>
                        <span className="text-sm font-bold">2.34%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">R² Score</span>
                        <span className="text-sm font-bold">0.87</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Training Data Size</span>
                        <span className="text-sm font-bold">365 days</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Model</span>
                        <span className="text-sm font-bold">Random Forest (100 trees)</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Portfolio Optimization Tab */}
        <TabsContent value="optimization" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Optimization</CardTitle>
              <CardDescription>
                Modern Portfolio Theory and optimization algorithms to maximize returns while minimizing risk
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="h-[400px]">
                <ChartContainer config={{}} className="h-full">
                  <Recharts.ScatterChart>
                    <Recharts.CartesianGrid strokeDasharray="3 3" />
                    <Recharts.XAxis 
                      type="number" 
                      dataKey="volatility" 
                      name="Volatility" 
                      domain={[0, 0.4]}
                      label={{ value: 'Volatility (Risk)', position: 'insideBottom', offset: -5 }}
                    />
                    <Recharts.YAxis 
                      type="number" 
                      dataKey="return" 
                      name="Expected Return" 
                      domain={[0, 0.4]}
                      label={{ value: 'Expected Return', angle: -90, position: 'insideLeft' }}
                    />
                    <Recharts.Tooltip 
                      formatter={(value) => `${(value as number * 100).toFixed(2)}%`}
                      labelFormatter={() => ""}
                    />
                    <Recharts.Scatter 
                      name="Portfolio" 
                      data={efficientFrontierData.filter(d => !d.portfolio)} 
                      fill="#8884d8"
                      opacity={0.5}
                    />
                    <Recharts.Scatter 
                      name="Current Portfolio" 
                      data={efficientFrontierData.filter(d => d.portfolio === 'Current')} 
                      fill="#ff0000" 
                      shape="circle"
                      opacity={1}
                    >
                      <Recharts.LabelList dataKey="portfolio" position="top" />
                    </Recharts.Scatter>
                    <Recharts.Scatter 
                      name="Optimized Portfolio" 
                      data={efficientFrontierData.filter(d => d.portfolio === 'Optimized')} 
                      fill="#00ff00" 
                      shape="circle"
                      opacity={1}
                    >
                      <Recharts.LabelList dataKey="portfolio" position="top" />
                    </Recharts.Scatter>
                  </Recharts.ScatterChart>
                </ChartContainer>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Current vs. Optimized Portfolio</CardTitle>
                    <CardDescription>
                      Comparison of current and ML-optimized asset allocations
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="h-[250px]">
                        <h3 className="text-sm font-medium mb-2 text-center">Current</h3>
                        <ChartContainer config={{}} className="h-full">
                          <Recharts.PieChart>
                            <Recharts.Pie
                              data={[
                                { name: "Bitcoin", value: 40 },
                                { name: "Ethereum", value: 25 },
                                { name: "Solana", value: 12 },
                                { name: "Cardano", value: 8 },
                                { name: "Others", value: 15 }
                              ]}
                              cx="50%"
                              cy="50%"
                              outerRadius={80}
                              dataKey="value"
                              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                              labelLine={false}
                            >
                              {assets.map((_, index) => (
                                <Recharts.Cell
                                  key={`cell-${index}`}
                                  fill={[
                                    "#8884d8",
                                    "#83a6ed",
                                    "#8dd1e1",
                                    "#82ca9d",
                                    "#a4de6c"
                                  ][index % 5]}
                                />
                              ))}
                            </Recharts.Pie>
                            <Recharts.Tooltip formatter={(value) => `${value}%`} />
                          </Recharts.PieChart>
                        </ChartContainer>
                      </div>
                      <div className="h-[250px]">
                        <h3 className="text-sm font-medium mb-2 text-center">Optimized</h3>
                        <ChartContainer config={{}} className="h-full">
                          <Recharts.PieChart>
                            <Recharts.Pie
                              data={[
                                { name: "Bitcoin", value: 30 },
                                { name: "Ethereum", value: 35 },
                                { name: "Solana", value: 15 },
                                { name: "Cardano", value: 5 },
                                { name: "Others", value: 15 }
                              ]}
                              cx="50%"
                              cy="50%"
                              outerRadius={80}
                              dataKey="value"
                              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                              labelLine={false}
                            >
                              {assets.map((_, index) => (
                                <Recharts.Cell
                                  key={`cell-${index}`}
                                  fill={[
                                    "#8884d8",
                                    "#83a6ed",
                                    "#8dd1e1",
                                    "#82ca9d",
                                    "#a4de6c"
                                  ][index % 5]}
                                />
                              ))}
                            </Recharts.Pie>
                            <Recharts.Tooltip formatter={(value) => `${value}%`} />
                          </Recharts.PieChart>
                        </ChartContainer>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Optimization Results</CardTitle>
                    <CardDescription>
                      Performance metrics comparison
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-center">
                          <div className="text-sm text-gray-500">Metric</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-gray-500">Current</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-gray-500">Optimized</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-sm font-medium">Expected Return</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm">20.00%</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-green-600 font-medium">25.00%</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-sm font-medium">Volatility</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm">25.00%</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-green-600 font-medium">22.00%</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-sm font-medium">Sharpe Ratio</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm">0.80</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-green-600 font-medium">1.14</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-sm font-medium">Max Drawdown</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm">32.50%</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-green-600 font-medium">27.80%</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-sm font-medium">Correlation (avg)</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm">0.65</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-green-600 font-medium">0.48</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Anomaly Detection Tab */}
        <TabsContent value="anomaly" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Anomaly Detection</CardTitle>
              <CardDescription>
                Statistical methods and clustering algorithms to identify unusual price movements
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="w-full md:w-1/4">
                  <Select value={selectedAsset} onValueChange={setSelectedAsset}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select Asset" />
                    </SelectTrigger>
                    <SelectContent>
                      {assets.map((asset) => (
                        <SelectItem key={asset.ticker} value={asset.name}>
                          {asset.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="w-full md:w-3/4">
                  <div className="flex flex-col space-y-2">
                    <label className="text-sm font-medium">
                      Anomaly Threshold (Z-score): {threshold.toFixed(1)}
                    </label>
                    <Slider
                      min={1.5}
                      max={4.0}
                      step={0.1}
                      value={[threshold]}
                      onValueChange={(value) => setThreshold(value[0])}
                    />
                  </div>
                </div>
              </div>

              <div className="h-[400px]">
                <ChartContainer config={{}} className="h-full">
                  <Recharts.ComposedChart data={anomalyData}>
                    <Recharts.CartesianGrid strokeDasharray="3 3" />
                    <Recharts.XAxis dataKey="date" />
                    <Recharts.YAxis />
                    <Recharts.Tooltip />
                    <Recharts.Legend />
                    <Recharts.Line
                      type="monotone"
                      dataKey="price"
                      stroke="#8884d8"
                      dot={false}
                      name={`${selectedAsset} Price`}
                    />
                    <Recharts.Scatter
                      dataKey={(data) => (data.isAnomaly ? data.price : null)}
                      fill="#ff0000"
                      name="Anomaly"
                    />
                  </Recharts.ComposedChart>
                </ChartContainer>
              </div>

              <div className="grid grid-cols-1 gap-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Z-Score Analysis</CardTitle>
                    <CardDescription>
                      Measuring deviation from normal price behavior
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[250px]">
                      <ChartContainer config={{}} className="h-full">
                        <Recharts.ComposedChart data={anomalyData}>
                          <Recharts.CartesianGrid strokeDasharray="3 3" />
                          <Recharts.XAxis dataKey="date" />
                          <Recharts.YAxis />
                          <Recharts.Tooltip />
                          <Recharts.Legend />
                          <Recharts.Line
                            type="monotone"
                            dataKey="zScore"
                            stroke="#8884d8"
                            dot={false}
                            name="Z-Score"
                          />
                          <Recharts.ReferenceLine
                            y={threshold}
                            stroke="red"
                            strokeDasharray="3 3"
                            label={{ value: 'Upper Threshold', position: 'right' }}
                          />
                          <Recharts.ReferenceLine
                            y={-threshold}
                            stroke="red"
                            strokeDasharray="3 3"
                            label={{ value: 'Lower Threshold', position: 'right' }}
                          />
                          <Recharts.ReferenceLine
                            y={0}
                            stroke="#666"
                            strokeDasharray="1 1"
                          />
                          <Recharts.Scatter
                            dataKey={(data) => (data.isAnomaly ? data.zScore : null)}
                            fill="#ff0000"
                            name="Anomaly"
                          />
                        </Recharts.ComposedChart>
                      </ChartContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Anomaly Statistics</CardTitle>
                    <CardDescription>
                      Summary of detected anomalies
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Total Anomalies Detected</span>
                        <span className="text-sm font-bold">{anomalyData.filter(d => d.isAnomaly).length}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Percentage of Anomalous Days</span>
                        <span className="text-sm font-bold">
                          {((anomalyData.filter(d => d.isAnomaly).length / anomalyData.length) * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Average Anomaly Z-Score</span>
                        <span className="text-sm font-bold">
                          {Math.abs(anomalyData.filter(d => d.isAnomaly).reduce((sum, item) => sum + item.zScore, 0) / 
                            Math.max(1, anomalyData.filter(d => d.isAnomaly).length)).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Largest Positive Anomaly</span>
                        <span className="text-sm font-bold text-green-600">+8.7%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Largest Negative Anomaly</span>
                        <span className="text-sm font-bold text-red-600">-9.2%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Detected Anomalies</CardTitle>
                    <CardDescription>
                      Most recent unusual price movements
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 max-h-[230px] overflow-y-auto">
                      {anomalyData
                        .filter(d => d.isAnomaly)
                        .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
                        .slice(0, 5)
                        .map((anomaly, index) => (
                          <div key={index} className="p-2 border rounded-md">
                            <div className="flex justify-between">
                              <span className="text-sm font-medium">{anomaly.date}</span>
                              <span className={`text-sm font-bold ${anomaly.zScore > 0 ? 'text-green-600' : 'text-red-600'}`}>
                                {anomaly.zScore > 0 ? '+' : ''}{(anomaly.zScore * 2).toFixed(2)}%
                              </span>
                            </div>
                            <div className="text-xs text-gray-500">
                              Z-Score: {anomaly.zScore.toFixed(2)} | Price: ${anomaly.price.toLocaleString(undefined, {maximumFractionDigits:2})}
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MLDashboard;
