
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AreaChart, BarChart, LineChart } from "@/components/ui/chart";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import {
  AreaChart as RechartsAreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart as RechartsBarChart,
  Bar,
  LineChart as RechartsLineChart,
  Line,
  Legend,
} from "recharts";
import { ArrowUpRight, TrendingUp, DollarSign, ArrowDown, ArrowUp, Wallet, BarChart2 } from "lucide-react";

// Mock data for the charts and portfolio
const portfolioData = [
  { name: "Bitcoin", value: 28000, allocation: 40, change: 5.2 },
  { name: "Ethereum", value: 15000, allocation: 25, change: -2.1 },
  { name: "Solana", value: 8000, allocation: 12, change: 8.7 },
  { name: "Cardano", value: 5000, allocation: 8, change: -1.3 },
  { name: "Polkadot", value: 4000, allocation: 7, change: 3.4 },
  { name: "Avalanche", value: 3000, allocation: 5, change: 6.8 },
  { name: "Chainlink", value: 2000, allocation: 3, change: -0.5 },
];

const performanceData = [
  { month: "Jan", value: 65000 },
  { month: "Feb", value: 68000 },
  { month: "Mar", value: 62000 },
  { month: "Apr", value: 70000 },
  { month: "May", value: 72000 },
  { month: "Jun", value: 68000 },
  { month: "Jul", value: 75000 },
  { month: "Aug", value: 82000 },
  { month: "Sep", value: 80000 },
  { month: "Oct", value: 85000 },
  { month: "Nov", value: 90000 },
  { month: "Dec", value: 95000 },
];

const marketData = [
  { date: "2023-01", btc: 16500, eth: 1200, sol: 12 },
  { date: "2023-02", btc: 22000, eth: 1600, sol: 20 },
  { date: "2023-03", btc: 28000, eth: 1800, sol: 22 },
  { date: "2023-04", btc: 30000, eth: 1900, sol: 25 },
  { date: "2023-05", btc: 27000, eth: 1750, sol: 20 },
  { date: "2023-06", btc: 29000, eth: 1850, sol: 24 },
  { date: "2023-07", btc: 31000, eth: 1950, sol: 26 },
  { date: "2023-08", btc: 28500, eth: 1800, sol: 23 },
  { date: "2023-09", btc: 26000, eth: 1650, sol: 21 },
  { date: "2023-10", btc: 34000, eth: 2000, sol: 30 },
  { date: "2023-11", btc: 36000, eth: 2100, sol: 35 },
  { date: "2023-12", btc: 42000, eth: 2300, sol: 45 },
];

const Index = () => {
  const [portfolioValue, setPortfolioValue] = useState(65000);
  const [gainPercentage, setGainPercentage] = useState(12.5);
  
  const exportData = () => {
    const dataStr = JSON.stringify(portfolioData, null, 2);
    const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
    
    const exportFileDefaultName = 'portfolio-data.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast({
      title: "Data exported successfully",
      description: "Your portfolio data has been exported as JSON",
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Fintech Portfolio Explorer</h1>
          <p className="text-gray-600 dark:text-gray-300">Track and analyze your cryptocurrency investments</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Portfolio Value</p>
                <h3 className="text-2xl font-bold">${portfolioValue.toLocaleString()}</h3>
              </div>
              <div className="h-12 w-12 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                <Wallet className="h-6 w-6 text-blue-600 dark:text-blue-300" />
              </div>
            </div>
          </Card>
          
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Gain/Loss</p>
                <h3 className="text-2xl font-bold flex items-center">
                  {gainPercentage > 0 ? (
                    <ArrowUp className="h-5 w-5 text-green-500 mr-1" />
                  ) : (
                    <ArrowDown className="h-5 w-5 text-red-500 mr-1" />
                  )}
                  {Math.abs(gainPercentage)}%
                </h3>
              </div>
              <div className="h-12 w-12 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center">
                <TrendingUp className="h-6 w-6 text-green-600 dark:text-green-300" />
              </div>
            </div>
          </Card>
          
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Assets</p>
                <h3 className="text-2xl font-bold">{portfolioData.length}</h3>
              </div>
              <div className="h-12 w-12 rounded-full bg-purple-100 dark:bg-purple-900 flex items-center justify-center">
                <BarChart2 className="h-6 w-6 text-purple-600 dark:text-purple-300" />
              </div>
            </div>
          </Card>
        </div>
        
        <Tabs defaultValue="portfolio" className="mb-8">
          <TabsList className="mb-6">
            <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="market">Market</TabsTrigger>
          </TabsList>
          
          <TabsContent value="portfolio" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4">Asset Allocation</h3>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsBarChart data={portfolioData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={80} />
                    <Tooltip 
                      formatter={(value) => [`${value}%`, 'Allocation']}
                      labelFormatter={(name) => `Asset: ${name}`}
                    />
                    <Legend />
                    <Bar dataKey="allocation" fill="#8884d8" name="Allocation (%)" />
                  </RechartsBarChart>
                </ResponsiveContainer>
              </div>
            </Card>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-xl font-semibold mb-4">Asset Values</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart data={portfolioData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value) => [`$${value}`, 'Value']}
                        labelFormatter={(name) => `Asset: ${name}`}
                      />
                      <Legend />
                      <Bar dataKey="value" fill="#82ca9d" name="Value (USD)" />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
              
              <Card className="p-6">
                <h3 className="text-xl font-semibold mb-4">24h Change</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart data={portfolioData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value) => [`${value}%`, '24h Change']}
                        labelFormatter={(name) => `Asset: ${name}`}
                      />
                      <Legend />
                      <Bar 
                        dataKey="change" 
                        fill={(entry) => (entry.change >= 0 ? "#82ca9d" : "#ff7782")}
                        name="24h Change (%)" 
                      />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>
            
            <Card className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold">Portfolio Assets</h3>
                <Button onClick={exportData}>Export Data</Button>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-4">Asset</th>
                      <th className="text-right py-3 px-4">Value (USD)</th>
                      <th className="text-right py-3 px-4">Allocation (%)</th>
                      <th className="text-right py-3 px-4">24h Change (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolioData.map((asset, index) => (
                      <tr key={index} className="border-b">
                        <td className="py-3 px-4">{asset.name}</td>
                        <td className="text-right py-3 px-4">${asset.value.toLocaleString()}</td>
                        <td className="text-right py-3 px-4">{asset.allocation}%</td>
                        <td className={`text-right py-3 px-4 ${asset.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          <span className="flex items-center justify-end">
                            {asset.change >= 0 ? (
                              <ArrowUp className="h-4 w-4 mr-1" />
                            ) : (
                              <ArrowDown className="h-4 w-4 mr-1" />
                            )}
                            {Math.abs(asset.change)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </TabsContent>
          
          <TabsContent value="performance" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4">Portfolio Performance</h3>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsAreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [`$${value}`, 'Portfolio Value']}
                      labelFormatter={(month) => `Month: ${month}`}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#8884d8" 
                      fill="#8884d8" 
                      fillOpacity={0.3} 
                      name="Portfolio Value (USD)"
                    />
                  </RechartsAreaChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </TabsContent>
          
          <TabsContent value="market" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4">Market Trends</h3>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsLineChart data={marketData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [`$${value}`, '']}
                      labelFormatter={(date) => `Date: ${date}`}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="btc" 
                      stroke="#F7931A" 
                      name="Bitcoin (USD)"
                      strokeWidth={2}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="eth" 
                      stroke="#627EEA" 
                      name="Ethereum (USD)"
                      strokeWidth={2}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="sol" 
                      stroke="#00FFA3" 
                      name="Solana (USD)"
                      strokeWidth={2}
                    />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
