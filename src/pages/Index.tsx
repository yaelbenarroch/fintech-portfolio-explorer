
import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChartContainer } from "@/components/ui/chart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import * as Recharts from "recharts";
import { 
  ArrowUpRight, 
  Wallet, 
  Brain, 
  LayoutDashboard, 
  ChartBar, 
  ChartLine,
  LineChart,
  Briefcase,
  AlertCircle
} from "lucide-react";

const Index = () => {
  // Sample data for the dashboard
  const portfolioData = [
    { name: "Bitcoin", value: 28000, allocation: 40 },
    { name: "Ethereum", value: 15000, allocation: 25 },
    { name: "Solana", value: 8000, allocation: 12 },
    { name: "Cardano", value: 5000, allocation: 8 },
    { name: "Polkadot", value: 4000, allocation: 7 },
    { name: "Avalanche", value: 3000, allocation: 5 },
    { name: "Chainlink", value: 2000, allocation: 3 },
  ];

  const performanceData = [
    { month: "Jan", Bitcoin: 42000, Ethereum: 3200 },
    { month: "Feb", Bitcoin: 44500, Ethereum: 3100 },
    { month: "Mar", Bitcoin: 47000, Ethereum: 3400 },
    { month: "Apr", Bitcoin: 45000, Ethereum: 3000 },
    { month: "May", Bitcoin: 49000, Ethereum: 3600 },
    { month: "Jun", Bitcoin: 52000, Ethereum: 3800 },
    { month: "Jul", Bitcoin: 50000, Ethereum: 3500 },
  ];

  const transactionData = [
    { name: "Buy Bitcoin", amount: 5000, date: "2023-10-15", type: "buy" },
    { name: "Sell Ethereum", amount: -2000, date: "2023-10-18", type: "sell" },
    { name: "Buy Solana", amount: 1500, date: "2023-10-20", type: "buy" },
    { name: "Sell Cardano", amount: -500, date: "2023-10-25", type: "sell" },
    { name: "Buy Polkadot", amount: 800, date: "2023-11-01", type: "buy" },
  ];

  const volatilityData = [
    { date: "2023-01", volatility: 0.65 },
    { date: "2023-02", volatility: 0.58 },
    { date: "2023-03", volatility: 0.70 },
    { date: "2023-04", volatility: 0.52 },
    { date: "2023-05", volatility: 0.48 },
    { date: "2023-06", volatility: 0.60 },
    { date: "2023-07", volatility: 0.55 },
  ];

  const totalValue = portfolioData.reduce((sum, item) => sum + item.value, 0);

  // ML Dashboard state
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
  const generatePredictionData = (days) => {
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
  const generateAnomalyData = (threshold) => {
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
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-purple-700 flex items-center">
          <Brain className="mr-2 h-6 w-6" />
          Fintech Portfolio & ML Analytics
        </h1>
        <p className="text-gray-600">
          Comprehensive cryptocurrency analysis and portfolio optimization with advanced ML models
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Portfolio Value</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${totalValue.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">+12.5% from last month</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Number of Assets</CardTitle>
            <ChartBar className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{portfolioData.length}</div>
            <p className="text-xs text-muted-foreground">Diversified portfolio</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Monthly Performance</CardTitle>
            <ChartLine className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">+8.2%</div>
            <p className="text-xs text-muted-foreground">Better than market average</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Volatility Index</CardTitle>
            <ArrowUpRight className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0.55</div>
            <p className="text-xs text-muted-foreground">Moderate risk level</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs containing Portfolio Overview and the 3 ML features */}
      <Tabs defaultValue="portfolio" className="w-full mb-8">
        <TabsList className="grid w-full grid-cols-4 mb-4">
          <TabsTrigger value="portfolio" className="flex items-center">
            <LayoutDashboard className="mr-2 h-4 w-4" /> Portfolio Overview
          </TabsTrigger>
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

        {/* Portfolio Overview Tab */}
        <TabsContent value="portfolio" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Asset Allocation</CardTitle>
                <CardDescription>Current portfolio distribution</CardDescription>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ChartContainer config={{}} className="h-full">
                  <Recharts.PieChart>
                    <Recharts.Pie
                      data={portfolioData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={(entry) => entry.name}
                      labelLine={true}
                    >
                      {portfolioData.map((entry, index) => (
                        <Recharts.Cell
                          key={`cell-${index}`}
                          fill={[
                            "#8884d8",
                            "#83a6ed",
                            "#8dd1e1",
                            "#82ca9d",
                            "#a4de6c",
                            "#d0ed57",
                            "#ffc658",
                          ][index % 7]}
                        />
                      ))}
                    </Recharts.Pie>
                    <Recharts.Tooltip />
                    <Recharts.Legend />
                  </Recharts.PieChart>
                </ChartContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Asset Performance</CardTitle>
                <CardDescription>Monthly trend for top assets</CardDescription>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ChartContainer config={{}} className="h-full">
                  <Recharts.LineChart data={performanceData}>
                    <Recharts.CartesianGrid strokeDasharray="3 3" />
                    <Recharts.XAxis dataKey="month" />
                    <Recharts.YAxis />
                    <Recharts.Tooltip />
                    <Recharts.Legend />
                    <Recharts.Line
                      type="monotone"
                      dataKey="Bitcoin"
                      stroke="#8884d8"
                      activeDot={{ r: 8 }}
                    />
                    <Recharts.Line type="monotone" dataKey="Ethereum" stroke="#82ca9d" />
                  </Recharts.LineChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Recent Transactions</CardTitle>
                <CardDescription>Last 5 portfolio transactions</CardDescription>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ChartContainer config={{}} className="h-full">
                  <Recharts.BarChart data={transactionData}>
                    <Recharts.CartesianGrid strokeDasharray="3 3" />
                    <Recharts.XAxis dataKey="name" />
                    <Recharts.YAxis />
                    <Recharts.Tooltip />
                    <Recharts.Legend />
                    <Recharts.Bar
                      name="Buy"
                      dataKey={(data) => (data.amount > 0 ? data.amount : 0)}
                      fill="#82ca9d"
                      radius={[4, 4, 0, 0]}
                    />
                    <Recharts.Bar
                      name="Sell"
                      dataKey={(data) => (data.amount < 0 ? -data.amount : 0)}
                      fill="#ff7782"
                      radius={[4, 4, 0, 0]}
                    />
                  </Recharts.BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Volatility Analysis</CardTitle>
                <CardDescription>Historical volatility trend</CardDescription>
              </CardHeader>
              <CardContent className="h-[300px]">
                <ChartContainer config={{}} className="h-full">
                  <Recharts.AreaChart data={volatilityData}>
                    <Recharts.CartesianGrid strokeDasharray="3 3" />
                    <Recharts.XAxis dataKey="date" />
                    <Recharts.YAxis />
                    <Recharts.Tooltip />
                    <Recharts.Area
                      type="monotone"
                      dataKey="volatility"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.3}
                    />
                  </Recharts.AreaChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

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

      <footer className="text-center text-gray-500 text-sm mt-8">
        <p>
          Data sourced from public APIs. This interactive dashboard incorporates advanced machine learning for cryptocurrency analysis and portfolio optimization.
        </p>
      </footer>
    </div>
  );
};

export default Index;
