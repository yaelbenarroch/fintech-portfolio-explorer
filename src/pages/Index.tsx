
import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChartContainer } from "@/components/ui/chart";
import * as Recharts from "recharts";
import { ArrowUpRight, BarChart3, LineChart, Wallet } from "lucide-react";

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

  // Function to determine bar color based on transaction type
  const getBarColor = (amount: number) => {
    return amount > 0 ? "#82ca9d" : "#ff7782";
  };

  return (
    <div className="container mx-auto p-4">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-purple-700">Fintech Portfolio Explorer</h1>
        <p className="text-gray-600">
          A comprehensive dashboard for cryptocurrency portfolio analysis and optimization
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
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{portfolioData.length}</div>
            <p className="text-xs text-muted-foreground">Diversified portfolio</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Monthly Performance</CardTitle>
            <LineChart className="h-4 w-4 text-muted-foreground" />
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
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
                {/* Split into two different bars for buy and sell transactions */}
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

      <div className="flex justify-center mb-8">
        <Button className="bg-purple-600 hover:bg-purple-700">
          View Full Streamlit Analysis
        </Button>
      </div>

      <footer className="text-center text-gray-500 text-sm">
        <p>
          Data sourced from public APIs. This dashboard is a simplified interface for the comprehensive
          Streamlit application.
        </p>
      </footer>
    </div>
  );
};

export default Index;
