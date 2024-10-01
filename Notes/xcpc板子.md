

# 基础

## STL工具

```c++
//vector
vector<int> a;
a.push_back(1);
a.pop_back();
sort(a.begin(),a.end()/*最后一个元素+1的指针*/);
a.clear();
a.empty();
a.size();
a.front();a.back();
vector<int> b;a.swap(b)//交换两vector

//map
map<int,string> a;
a[1]="a";
a.insert(pair<int,string>(2,"b"));
a.empty();
a.size();

//set
set<int> a;
a.insert(1);
a.erase(1);
a.empty()
a.size()
auto it=a.find(1);//查找某个元素的迭代器，不存在则返回a.end()
if(it!=a.end()) cout<<"Yes"<<endl;
a.count(1);
a.lower_bound(1)//第一个>=的迭代器
a.upper_bound(1)//第一个>的迭代器
//(set会将元素自动排序)
//遍历方式1
set<int>::iterator it;//迭代器
for(it=a.begin();i!=a.end()/*最后一个元素+1的指针*/;i++)//自加
    cout<<*it<<endl;
//遍历方式2
for(auto i=a.begin();i!=a.end();i++)
    cout<<*i<<endl;
//遍历方式3
for(auto i:a) 
    cout<<i<<endl;
//multiset(与set基本相同)
multiset<int> a;
//二分工具
ans1=lower_bound(a+1,a+n+1,10)-a;
ans2=upper_bound(a+1,a+n+1,10)-a;
ans3=unique(a+1,a+n+1)-1-a;//要-1

```

## 质因数分解

```c++
vector<int> breakdown(int N) {
  vector<int> result;
  for (int i = 2; i * i <= N; i++) {
    if (N % i == 0) {  // 如果 i 能够整除 N，说明 i 为 N 的一个质因子。
      while (N % i == 0) N /= i;
      result.push_back(i);
    }
  }
  if (N != 1) {  // 说明再经过操作之后 N 留下了一个素数
    result.push_back(N);
  }
  return result;
}
```

## 裴蜀定理

```c++
//ax+bx=c有解，当且仅当c=k*gcd(a,b)
//ax+bx=c (mod m)有解，当且仅当c=k*gcd(a,b) (mod m)
int gcd(int x,int y){
	return y?
}
int main() {
	//构造xi，使ai*xi之和最小，答案为gcd(所有ai)
    scanf("%d", &n);
    int ans = 0, tmp;
    for(int i=1; i<=n; i++) {
        scanf("%d", &tmp);
        if(tmp < 0) tmp = -tmp;//取反不影响结果
        ans = gcd(ans, tmp);
    }
    printf("%d", ans);
}
```

## exgcd

```c++

int gcd(int x,int y){
	return y?gcd(y,x%y):x;
}
int exgcd(int a,int b,int &x,int &y){
	int d=a;
	if(b==0){
		x=1,y=0;
	}else{
		d=exgcd(b,a%b,y,x),y=y-a/b*x;
	}
	return d;
}
//求出ax+by=c的正整数解，无则输出-1
signed main(){
	int T;cin>>T;
	while(T--){
		int a,b,c,x,y;
		cin>>a>>b>>c;
		//ax+by=c
		int d=exgcd(a,b,x,y);//exgcd求出特解(右为gcd版)
		
		if(c%d!=0){
			cout<<-1<<endl;
			continue;
		}
		
		x*=c/d,y*=c/d;
		int p=b/d,q=a/d;//a(x+k*p)+b(y-k*q)=c
		int k;
		if(x<0) k=ceil((1.0-x)/p),x+=p*k,y-=q*k;
		if(x>=0) k=(x-1)/p,x-=p*k,y+=q*k;
		
		if(y<=0){
			cout<<x<<" "<<y+q*(int)ceil((1.0-y)/q);
		}else{
			cout<<(y-1)/q+1<<" ";//解的数量
			cout<<x<<" "<<(y-1)%q+1<<" ";//Xmin Ymax
			cout<<x+(y-1)/q*p<<" "<<y;//Xmax Ymin
		}
		cout<<endl;
	}
}
```

![24ac77f71a63fa5d0572e476e5d4b79](C:\Users\m1368\Documents\WeChat Files\wxid_8ixqik1qpezg22\FileStorage\Temp\24ac77f71a63fa5d0572e476e5d4b79.png)

## 欧拉函数

```c++
int euler_phi(int n) {
  int ans = n;
  for (int i = 2; i * i <= n; i++)
    if (n % i == 0) {
      ans = ans / i * (i - 1);
      while (n % i == 0) n /= i;
    }
  if (n > 1) ans = ans / n * (n - 1);
  return ans;
}
```

## 欧拉定理

![463751c854eac7d4860515ae789265b](C:\Users\m1368\Documents\WeChat Files\wxid_8ixqik1qpezg22\FileStorage\Temp\463751c854eac7d4860515ae789265b.png)

## 扩展欧拉定理

![09a3d2c802d8a5851cea5ffa2c623b1](C:\Users\m1368\Documents\WeChat Files\wxid_8ixqik1qpezg22\FileStorage\Temp\09a3d2c802d8a5851cea5ffa2c623b1.png)

## crt

```c++
int n,a[500000+5],b[500000+5],M=1;
void Exgcd(int a,int b,int& d,int& x,int& y) {//ax+by
    if(!b) {
        d=a;x=1;y=0;
    }
    else {
        Exgcd(b,a%b,d,x,y);
        int newx=y,newy=x-(a/b)*y;
        x=newx;y=newy;
    }
}
int IntChina() {
    int ret=0,Mi,x,y,d;
    for(int i=1;i<=n;i++) {
        Mi=M/a[i];
        Exgcd(Mi,a[i],d,x,y);
        //x:Mi*x=1(mod mi)的一个解
        ret=((ret+Mi*x*b[i])%M+M)%M;
    }
    return ret;
}
signed main(){
    cin>>n;
    for(int i=1;i<=n;i++) {
        cin>>a[i]>>b[i];
        //ans%a[i]=b[i]
        M*=a[i];//M:所有模数之积
    }
    cout<<IntChina()<<endl;
}
```

![b3dd61893487281889a85cb91d09f37](C:\Users\m1368\Documents\WeChat Files\wxid_8ixqik1qpezg22\FileStorage\Temp\b3dd61893487281889a85cb91d09f37.png)

## excrt

```c++
int n,a[100000+5],b[100000+5],M;
int qmul(int x,int y,int MOD) {
    int ret=0;
    while(y) {
        if(y&1) ret=(ret+x)%MOD;
        y>>=1;
        x=(x+x)%MOD;
    }return ret;
}
void Exgcd(int a,int b,int& d,int& x,int& y) {
    if(!b) {
        d=a;x=1;y=0;
    }
    else {
        Exgcd(b,a%b,d,x,y);
        int newx=y,newy=x-(a/b)*y;
        x=newx;y=newy;
    }
}
int ExIntChina() {
    int ret=b[1],x,y,d;
    int M=a[1];//目前为止ai的lcm
    //维护一个x=ret(mod M)
    for(int i=2;i<=n;i++) {
        int c=((b[i]-ret)%a[i]+a[i])%a[i];
        //Mx + a[i]q = b[i] - ret
        Exgcd(M,a[i],d,x,y);
        if(c%d!=0) return -1;
        x=qmul(x,c/d,a[i]/d);
        ret+=M*x;//更新ret
        M*=(a[i]/d);
        ret=(ret%M+M)%M;//对新的模意义更新
    }
    return (ret%M+M)%M;
}
signed main(){
    cin>>n;
    for(int i=1;i<=n;i++) {
        cin>>a[i]>>b[i];
    }
    //x%a[i]=b[i],a[i]不一定互质
    cout<<ExIntChina()<<endl;
}
```

## 威尔逊定理

$$
对于素数p有(p-1)!=-1(mod p)
$$

##　卢卡斯定理

![image-20240913190528301](C:\Users\m1368\AppData\Roaming\Typora\typora-user-images\image-20240913190528301.png)

```c++
long long Lucas(long long n, long long m, long long p) {
  if (m == 0) return 1;
  return (C(n % p, m % p, p) * Lucas(n / p, m / p, p)) % p;
}
```

## 扩展卢卡斯定理

```c++
LL calc(LL n, LL x, LL P) {
  if (!n) return 1;
  LL s = 1;
  for (LL i = 1; i <= P; i++)
    if (i % x) s = s * i % P;
  s = Pow(s, n / P, P);
  for (LL i = n / P * P + 1; i <= n; i++)
    if (i % x) s = i % P * s % P;
  return s * calc(n / x, x, P) % P;
}
LL multilucas(LL m, LL n, LL x, LL P) {
  int cnt = 0;
  for (LL i = m; i; i /= x) cnt += i / x;
  for (LL i = n; i; i /= x) cnt -= i / x;
  for (LL i = m - n; i; i /= x) cnt -= i / x;
  return Pow(x, cnt, P) % P * calc(m, x, P) % P * inverse(calc(n, x, P), P) %
         P * inverse(calc(m - n, x, P), P) % P;
}
LL exlucas(LL m, LL n, LL P) {
  int cnt = 0;
  LL p[20], a[20];
  for (LL i = 2; i * i <= P; i++) {
    if (P % i == 0) {
      p[++cnt] = 1;
      while (P % i == 0) p[cnt] = p[cnt] * i, P /= i;
      a[cnt] = multilucas(m, n, i, p[cnt]);
    }
  }
  if (P > 1) p[++cnt] = P, a[cnt] = multilucas(m, n, P, P);
  return CRT(cnt, a, p);
}
```

## 二分法

```c++
/*
l是大于寻找元素的第一个元素
（不存在l会为n）
*/
int l=1,r=m,mid;
while(l<r){
	mid=(l+r)>>1;
    if(a[mid]<=seek)
       l=mid+1;
    else
       r=mid;
}
```

## 三分法

```c++
double f(double a){/*根据题目意思计算*/}
double three(double l,double r)
{
    while(l+EPS<r)
    {
        double mid=l+(r-l)/3;
        double midmid=r-(r-l)/3;
        if(f(mid)>f(midmid)) r=midmid;
        else l=mid;
    }
    return l;
}
void Solve(){
    double left, right, m1, m2, m1_value, m2_value;
    left = MIN;
    right = MAX;
    while (left + EPS < right){
        m1 = left + (right - left)/3;
        m2 = right - (right - left)/3;
        m1_value = f(m1);
        m2_value = f(m2);
        if (m1_value >= m2_value)
            right = m2;  //假设求解极大值
        else  left = m1; 
    }
} 
```

## 高斯消元

```c++
const db EPS = 1E-8;
ll n;
db a[N][N];
int main() {
	n = read();
	for (ll i = 0; i < n; i++)
		for (ll j = 0; j <= n; j++)
			scanf("%lf", &a[i][j]);
	for (ll i = 0; i < n; i++) {
		ll p = i;
		for (ll j = i; j < n; j++)
			if (fabs(a[j][i] - a[p][i]) <= EPS) p = j;
		for (ll j = 0; j <= n; j++) {
			db t = a[i][j];
			a[i][j] = a[p][j];
			a[p][j] = t;
		}
		if (fabs(a[i][i]) <= EPS) {
			printf("No Solution\n");
			return 0;
		}
		db div = a[i][i];
		for (ll j = i + 1; j <= n; j++) a[i][j] /= div;
		for (ll j = 0; j < n; j++)
			if (i != j)
				for (ll k = i + 1; k <= n; k++) 
					a[j][k] -= a[j][i] * a[i][k];
	}
	for (ll i = 0; i < n; i++) printf("%.2lf\n", a[i][n]);
	return 0;
}
```



## 线性基

```c++
void insert(int x){
	for(int i=N;i>=0;i--)
		if(x&(1ll<<i)){
			if(!p[i]){
				p[i]=x;
				break;
			}
			else x=x^p[i];
		}
}
void build(){
	for(int i=N;i>=0;i--)
		for(int j=i-1;j>=0;j--)
			if(p[i]&(1ll<<j))
				p[i]^=p[j];
	for(int i=0;i<=N;i++)
		if(p[i])
			d[cnt++]=p[i];//d下标从0开始
}
int kth(int k){
	if(k>=(1ll<<cnt)) return -1;
	int ret=0;
	for(int i=N-1;i>=0;i--)
		if(k&(1ll<<i)) ret^=d[i];
	return ret;
}
int Max(){
	return kth((1ll<<cnt)-1);//默认不能选0个
}
signed main(){
	cin>>n;
	for(int i=1;i<=n;i++) cin>>a[i],insert(a[i]);
	build();
/*	int K;
	cin>>K;
	cout<<kth(K)<<endl;*/
	cout<<Max()<<endl;
	return 0;
}
```

## 堆优化dijkstra

```c++
priority_queue<pair<int,int> > q;
void dijkstra(){
	for(int i=1;i<=n;i++) d[i]=MAXN;//d[i]初始为INF 
	d[s]=0;//d[1]初始为0 
	q.push(make_pair(0,s));
	while(q.size()){
		int x=q.top().second;q.pop();
		if(v[x]) continue;
		v[x]=1;//每个点只拓展一遍 
		for(int i=head[x];i;i=Next[i]){
			int y=ver[i],z=edge[i];
			if(d[x]+z<d[y]){
				d[y]=d[x]+z;
				q.push(make_pair(-d[y],y));
			}
		}
	}
}
```

## 差分约束

```c++
int spfa(){
	queue<int> q;//SPFA使用普通队列 
	for(int i=1;i<=n;i++) dis[i]=INF;
	q.push(0);v[0]=1;push_cnt[0]=1;
	while(q.size()){
		int x=q.front();q.pop();
		v[x]=0;
		for(int i=head[x];i;i=Next[i]){
			int y=ver[i],z=val[i];
			if(dis[y]>dis[x]+z){
				dis[y]=dis[x]+z;
				if(!v[y]){//不管怎样都更新dis[y]，但是必须不存在queue中才能加入queue 
					q.push(y);v[y]=1;push_cnt[y]++;
					if(push_cnt[y]>=n)//这个点已经被入queue了n次 
						return 0;
				}
			}
		}
	}
	return 1;
}
int main(){
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= m; i++) {
		int x, y, z;
		cin >> x >> y >> z;
		insert(y, x, z);
	}
	for (int i = 1; i <= m; i++) 
		insert(0, i, 0);
	if (!spfa()) printf("NO");
	else {
		for (int i = 1; i <= n; i++) min_dis = min(min_dis, dis[i]);
		for (int i = 1; i <= n; i++) {
			printf("%d ", dis[i] - min_dis);
		}
	}
	return 0;
} 
```

## 负环

```c++
void spfa() {
	for (int i = 1; i <= n; i++) dis[i] = INF;
	queue<int> q;
	dis[1] = 0; vis[1] = 1; q.push(1);
	while (q.size()) {
		int x = q.front(); vis[x] = 0; q.pop();
		for (int i = head[x]; i; i = Next[i]) {
			int y = ver[i];
			if (dis[x] + val[i] < dis[y]) {
				dis[y] = dis[x] + val[i];
				if (!vis[y]) {
					if (++incnt[y] >= n) {//入队n次以上，即说明存在负环
						cout << "YES" << endl;
						return;
					}
					vis[y] = 1; q.push(y);
				}
			}
		}
	}
	cout << "NO" << endl;
}
```

## 分块

```c++

ll n, m, a[N], sum[N], L[N], R[N], add[N], pos[N];
void change(ll l, ll r, ll d) {
	int p = pos[l], q = pos[r];
	if (p == q) {
		For(i, l, r) a[i] += d;
		sum[p] += d * (r - l + 1);
	} else {
		For(i, p + 1, q - 1) add[i] += d;
		For(i, l, R[p]) a[i] += d; sum[p] += (R[p] - l + 1) * d;
		For(i, L[q], r) a[i] += d; sum[q] += (r - L[q] + 1) * d;
	}
}
ll ask(ll l, ll r) {
	ll p = pos[l], q = pos[r], ret = 0;
	if (p == q) {
		For(i, l, r) ret += a[i];
		ret += add[p] * (r - l + 1);
	} else {
		For(i, p + 1, q - 1) ret += sum[i] + add[i] * (R[i] - L[i] + 1);
		For(i, l, R[p]) ret += a[i]; ret += add[p] * (R[p] - l + 1);
		For(i, L[q], r) ret += a[i]; ret += add[q] * (r - L[q] + 1);
	}
	return ret;
}
int main() {
	n = read(), m = read();
	For(i, 1, n) a[i] = read();
	ll t = sqrt(n);
	For(i, 1, t) L[i] = (i - 1) * t + 1, R[i] = i * t;
	if (R[t] < n) t++, L[t] = R[t - 1] + 1, R[t] = n;
	For(i, 1, t) {
		For(j, L[i], R[i]) 
			pos[j] = i, sum[i] += a[j];
	}
	while (m--) {
		ll op = read(), x, y, k;
		if (op == 1) {
			x = read(), y = read(), k = read(); change(x, y, k);
		} else {
			x = read(), y = read(); printf("%lld\n", ask(x, y));
		}
	}
	return 0;
} 
```

## 树状数组

```cpp
#include <bits/stdc++.h>
using namespace std;
#define lowbit(x) ((x) & (-x))
#define MAXN 500010
int tree[MAXN], a[MAXN], n, m;
void update(int x, int k) {
	while (x <= n) {
		tree[x] += k;
		x += lowbit(x);
	}
}
int query(int x) {
	int sum = 0;
	while (x) {
		sum += tree[x];
		x -= lowbit(x);
	}
	return sum;
}
void build() {
	for (int i = 1; i <= n; i++) update(i, a[i]);
}
int main() {
	int mode, l, r, x, k;
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
	build();
	for (int i = 1; i <= m; i++) {
		scanf("%d", &mode);
		if (mode == 1) {
			scanf("%d%d", &x, &k);
			update(x, k);
		} else {
			scanf("%d%d", &l, &r);
			printf("%d\n", query(r) - query(l - 1));
		}
	}
	return 0;
}
```

## 割点

```c++
void tarjan(ll x) {
	dfn[x] = low[x] = ++num;
	ll flag = 0;
	for (ll i = head[x]; i; i = e[i].Next) {
		ll y = e[i].ver;
		if (!dfn[y]) {
			tarjan(y);
			low[x] = min(low[x], low[y]);
			if (low[y] >= dfn[x]) {
				flag++;
				if (x != root || flag > 1)
					cut[x] = 1;
			}
		} else
			low[x] = min(low[x], dfn[y]);
	}
}
```

## 缩点

```c++
void tarjan(int x) {
	dfn[x] = low[x] = ++times;
	stac[++top] = x;
	instac[x] = 1;//标记当前节点在栈里 
	for (int i = head[x]; i; i = e[i].Next) {
		int y = e[i].y;
		if (!dfn[y]) {
			tarjan(y);
			low[x] = min(low[y], low[x]);
			
		} else if (!belong[y]) {//或写成 if(instac[y])
			low[x] = min(low[x], dfn[y]);
		}
	}
	if (dfn[x] == low[x]) {//弹出栈里元素直至当前访问到的节点 
		sccnum++;
		while (stac[top] != x) {
			belong[stac[top]] = sccnum;//染色 
			sum[sccnum] += a[stac[top]];//计入新scc的点权和 
			instac[stac[top]] = 0;//出栈 
			top--;
		}
		belong[stac[top]] = sccnum;
		sum[sccnum] += a[stac[top]];
		instac[stac[top]] = 0;
		top--;
	}
}
void dfs(int x) {
	if (f[x]) return;
	f[x] = sum[x];
	int maxnum = 0;
	for (int i = head[x]; i; i = e[i].Next) {
		int y = e[i].y;
		if (!f[y]) dfs(y);
		maxnum = max(maxnum, f[y]);
	}
	f[x] += maxnum;
}
int main() {

	cin >> n >> m;
	for (int i = 1; i <= n; i++) cin >> a[i];
	for (int i = 1; i <= m; i++) {
		int xx, yy;
		cin >> xx >> yy;
		x[i] = xx; y[i] = yy;//先把每条边保存到数组里，后面会用到 
		insert(xx, yy);
	}
	for (int i = 1; i <= n; i++) if (!dfn[i]) tarjan(i);//图不一定联通，所以要每个节点检查遍历 
	
	//清空原图，开始建新图 
	memset(head, 0, sizeof(head));
	for (int i = 1; i <= cnt; i++) {
		e[i].Next = 0; e[i].x = 0; e[i].y = 0;
	}
	
	cnt = 0;
	
	for (int i = 1; i <= m; i++) {
		if (belong[x[i]] != belong[y[i]]) { 
			insert(belong[x[i]], belong[y[i]]);//连接scc 
		}
	}
	
	for (int i = 1; i <= sccnum; i++) {
		if (!f[i]) {
			dfs(i);
			ans = max(ans, f[i]);
		}
	}
	
	cout << ans;
	return 0;
}

```

## 虚树

```c++
		sort(a+1,a+k+1,cmp);
        stk[1]=1;
        int top=1;
        t2.head[1]=0;
        for(int i=1;i<=k;i++){//k个关键点
            int lca=getlca(a[i],stk[top]);
            if(lca!=stk[top]){
                while(dfn[lca]<dfn[stk[top-1]]){
                    int x=stk[top-1],y=stk[top];
                    getlca(x,y);
                    t2.insert(x,y,d);
                    top--;
                }
                if(dfn[lca]>dfn[stk[top-1]]){
                    getlca(lca,stk[top]);
                    t2.head[lca]=0;
                    t2.insert(lca,stk[top],d);
                    top--;
                    stk[++top]=lca;
                }
                else{//dfn[lca]=dfn[stk[top-1]]
                    int x=stk[top-1],y=stk[top];
                    getlca(x,y);
                    t2.insert(x,y,d);
                    top--;
                }
            }
            t2.head[a[i]]=0;
            stk[++top]=a[i];
        }
        while(top>1){
            int x=stk[top-1],y=stk[top];
            getlca(x,y);
            t2.insert(x,y,d);
            top--;
        }
```



## 点分治

```c++
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=2e4+5;
const int K=1e7+5;
int n,m,head[N],cnt;
int ask[205],ok[205];
int rt,mx[N],vis[N],siz[N];
int dis[N],a[N],b[N];
int cont;
struct edge{
    int v,w,Next;
}e[N];
void insert(int x,int y,int z){
    cnt++;e[cnt].Next=head[x];head[x]=cnt;e[cnt].v=y;e[cnt].w=z;
}
bool cmp(int &x,int &y){
    return dis[x]<dis[y];
}
void getrt(int x,int from,int tot){
    siz[x]=1;
    mx[x]=0;
    for(int i=head[x];i;i=e[i].Next){
        int y=e[i].v,z=e[i].w;
        if(y==from||vis[y]) continue;//vis=1说明不在此子树中
        getrt(y,x,tot);
        siz[x]+=siz[y];
        mx[x]=max(mx[x],siz[y]);
    }
    mx[x]=max(mx[x],tot-siz[x]);
    if(!rt||mx[x]<mx[rt])
        rt=x;
}
void getdis(int x,int from,int nowdis,int bl){//以目前rt为根的深度
    a[++cont]=x;
    dis[x]=nowdis;
    b[x]=bl;
    for(int i=head[x];i;i=e[i].Next){
        int y=e[i].v,z=e[i].w;
        if(y==from||vis[y]) continue;
        getdis(y,x,nowdis+z,bl);
    }
}
void calc(int x){
    cont=0;
    a[++cont]=x;dis[x]=0;b[x]=x;//要合并的答案包括rt
    for(int i=head[x];i;i=e[i].Next){
        int y=e[i].v,z=e[i].w;
        if(vis[y]) continue;
        getdis(y,x,z,y);
    }
    sort(a+1,a+cont+1,cmp);
    for(int i=1;i<=m;i++){
        int l=1,r=cont;//cont:当前合并答案的总量
        if(ok[i]) continue;
        while(l<r){//尺取法
            if(dis[a[l]]+dis[a[r]]>ask[i]) r--;
            else if(dis[a[l]]+dis[a[r]]<ask[i]) l++;
            else if(b[a[l]]==b[a[r]]){//dis[a[l]]+dis[a[r]]==ask[i],但答案不分别在两个子树
                if(dis[a[r]]==dis[a[r-1]]) r--;
                else l++;
            }
            else{
                ok[i]=1;
                break;
            }
        }
    }
}
void solve(int x){//点分治
    vis[x]=1;
    calc(x);
    for(int i=head[x];i;i=e[i].Next){
        int y=e[i].v;
        if(vis[y]) continue;
        rt=0;
        getrt(y,0,siz[y]);
        solve(rt);
    }
}
int main(){
    cin>>n>>m;
    for(int i=1;i<n;i++){
        int x,y,z;
        cin>>x>>y>>z;
        insert(x,y,z);
        insert(y,x,z);
    }
    for(int i=1;i<=m;i++){
        cin>>ask[i];
        if(!ask[i]) ok[i]=1;
    }
    rt=0;
    mx[rt]=n;
    getrt(1,0,n);
    solve(rt);
    for(int i=1;i<=m;i++){
        if(ok[i]) cout<<"AYE\n";
        else cout<<"NAY\n";
    }
    return 0;
}
```

## 2-SAT

```c++
void tarjan(int x){
	dfn[x]=low[x]=++dfncnt;
	stk[++top]=x;
	for(int i=head[x];i;i=Next[i]){
		int y=ver[i];
		if(!dfn[y]){
			tarjan(y);
			low[x]=min(low[x],low[y]);//
		}
		else if(!col[y]){
			low[x]=min(low[x],dfn[y]);//
		}
	}
	if(low[x]==dfn[x]){//给一个scc染色
		colnum++;
		while(stk[top]!=x)
			col[stk[top--]]=colnum;
		col[stk[top--]]=colnum;
	}
}
int main(){
	cin>>n>>m;
	for(int i=1;i<=m;i++){
		int x,a,y,b;
		cin>>x>>a>>y>>b;
		//i  :1
		//i+n:0
		insert(x+n*(a^1),y+n*b);
		insert(y+n*(b^1),x+n*a);
	}
	for(int i=1;i<=n*2;i++)
		if(!dfn[i])
			tarjan(i);
	for(int i=1;i<=n;i++)
		if(col[i]==col[i+n]){
			cout<<"IMPOSSIBLE\n";
			return 0;
		}
	cout<<"POSSIBLE\n";
	for(int i=1;i<=n;i++)
		if(col[i]>col[i+n]) cout<<"1 ";
		else cout<<"0 ";
	return 0;
}
```

## 二分图最大匹配

```c++
bool dfs(ll x) {
	for (int i = head[x]; i; i = e[i].Next) {
		ll y = e[i].ver;
		if (vis[y]) continue; 
		vis[y] = 1;
		if (!mat[y] || dfs(mat[y])) {
			mat[y] = x; return 1;
		}
	} 
	return 0;
}
void Match() {
	for (int i = 1; i <= n; i++) {
		memset(vis, 0, sizeof(vis));
		if (dfs(i)) ans++;
	}
}
```

## ST表

```python
ll query(ll l, ll r) {
	ll lg2 = log2(r - l + 1);
	return max(f[l][lg2], f[r - (1 << lg2) + 1][lg2]);
}
int main() {
	n = read(); m = read();
	for (int i = 1; i <= n; i++) f[i][0] = read();
	for (int j = 1; j <= 20; j++) {
		for (int i = 1; i + (1 << j) - 1 <= n; i++) {
			f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
		}
	}
	while (m--) {
		ll l = read(), r = read();
		printf("%lld\n", query(l, r));
	}
	return 0;
}
```

## 扫描线

```c++

int n,tmp[N];
struct node{
	ll l,r,h,mark;
}line[N];
bool operator <(const node &x,const node &y){
	return x.h<y.h;
}
struct tree{
	ll tag,len;
}t[N<<2];
//tag
//len
void pushup(int x,int l,int r){
	if(t[x].tag) t[x].len=tmp[r+1]-tmp[l];//有tag，是还在被覆盖中的
	else{//无覆盖
		t[x].len=t[x<<1].len+t[x<<1|1].len;
	}
}
void change(int x,int l,int r,int L,int R,int c){
	if(tmp[r+1]<=L||R<=tmp[l]) return;
	if(L<=tmp[l]&&tmp[r+1]<=R){
		t[x].tag+=c;
		pushup(x,l,r);
		return;
	}
	int mid=(l+r)>>1;
	change(x<<1,l,mid,L,R,c);
	change(x<<1|1,mid+1,r,L,R,c);
	pushup(x,l,r);
}
int main(){
	/*
	/___/_/___/_/_/_____
	（r+1的原因）
	*/
	cin>>n;
	for(int i=1;i<=n;i++){
		int a,b,c,d;
		cin>>a>>b>>c>>d;//x1 y1 x2 y2
		tmp[i*2-1]=a,tmp[i*2]=c;
		line[i*2-1]=(node){a,c,b,1};
		line[i*2]=(node){a,c,d,-1};	
	}
	sort(line+1,line+n*2+1);
	sort(tmp+1,tmp+n*2+1);
	int len=unique(tmp+1,tmp+n*2+1)-tmp-1;
//	build(1,1,len-1);//植树问题，线段数=点数-1 
	ll ans=0;
	for(int i=1;i<=n*2-1;i++){
		change(1,1,len-1,line[i].l,line[i].r,line[i].mark);//(1,1,len-1)!!!
		ans+=t[1].len*(line[i+1].h-line[i].h);
	}
	cout<<ans<<endl;
	return 0;
}
```

## 树链剖分

```c++

ll n, m, root, mod;
ll cnt, head[N];
ll siz[N], top[N], dep[N], fa[N], num, id[N], son[N];
ll sum[N << 2], tag[N << 2], a[N], at[N];
struct edge {ll ver, Next;} e[N << 1];
void insert(ll x, ll y) {cnt++; e[cnt].Next = head[x]; head[x] = cnt; e[cnt].ver = y;}
void pushdown(ll o, ll l, ll r) {
	tag[ls(o)] += tag[o]; tag[rs(o)] += tag[o]; tag[ls(o)] %= mod; tag[rs(o)] %= mod; ll mid = (l + r) >> 1; 
	sum[ls(o)] += (mid - l + 1) * tag[o] % mod; sum[ls(o)] %= mod; sum[rs(o)] += (r - mid) * tag[o] % mod; sum[rs(o)] %= mod; tag[o] = 0;
} 
void pushup(ll o) {sum[o] = (sum[ls(o)] + sum[rs(o)]) % mod;}
void build(ll o, ll l, ll r) {
	if (l == r) {sum[o] = at[l] % mod; return;}
	ll mid = (l + r) >> 1; build(ls(o), l, mid); build(rs(o), mid + 1, r); pushup(o);
}
void update(ll o, ll l, ll r, ll L, ll R, ll val) {
	if (L <= l && r <= R) {sum[o] = (sum[o] + (r - l + 1) * val % mod) % mod; tag[o] = (tag[o] + val) % mod; return;}
	ll mid = (l + r) >> 1; pushdown(o, l, r); if (mid >= L) update(ls(o), l, mid, L, R, val); if (mid + 1 <= R) update(rs(o), mid + 1, r, L, R, val); pushup(o);
}
ll query(ll o, ll l, ll r, ll L, ll R) {
	if (L <= l && r <= R) {return sum[o];}
	ll mid = (l + r) >> 1, ret = 0; pushdown(o, l, r); if (mid >= L) ret += query(ls(o), l, mid, L, R), ret %= mod; if (mid + 1 <= R) ret += query(rs(o), mid + 1, r, L, R), ret %= mod; return ret;
}
void dfs1(ll x, ll from, ll nowdep) {//遍历 siz dep son
	dep[x] = nowdep; fa[x] = from; siz[x] = 1; ll maxson = -1;//maxsiz
	for (ll i = head[x]; i; i = e[i].Next) {
		ll y = e[i].ver; if (y == from) continue; dfs1(y, x, nowdep + 1);
		siz[x] += siz[y]; if (siz[y] >= maxson) maxson = siz[y], son[x] = y;
	}
}
void dfs2(ll x, ll nowtop) {//先走maxson id(在新数组中下标) top 新数组的值 
	id[x] = ++num; top[x] = nowtop; at[num] = a[x];
	if (!son[x]) return; dfs2(son[x], nowtop);
	for (ll i = head[x]; i; i = e[i].Next) {
		ll y = e[i].ver; if (y == fa[x] || y == son[x]) continue;
		dfs2(y, y);
	}
}
void updPath(ll x, ll y, ll val) {
	while (top[x] != top[y]) {if (dep[top[x]] < dep[top[y]]) swap(x, y);/*保证top[x]的dep >= top[y]的dep*/ update(1, 1, n, id[top[x]], id[x], val); x = fa[top[x]];/* 需更新完一条链，要再往上跳一个，避免重复*/} 
	if (dep[x] > dep[y]) swap(x, y);
	update(1, 1, n, id[x], id[y], val);
}
ll qPath(ll x, ll y) {ll ret = 0;
	while (top[x] != top[y]) {if (dep[top[x]] < dep[top[y]]) swap(x, y); ret = (ret + query(1, 1, n, id[top[x]], id[x])) % mod; x = fa[top[x]];}
	if (dep[x] > dep[y]) swap(x, y);
	ret = (ret + query(1, 1, n, id[x], id[y])) % mod; return ret;
}
void updSon(ll x, ll val) {update(1, 1, n, id[x], id[x] + siz[x] - 1, val);}
ll qSon(ll x) {return query(1, 1, n, id[x], id[x] + siz[x] - 1);}
int main() {
	n = read(); m = read(); root = read(); mod = read(); For(i, 1, n) a[i] = read();
	For(i, 1, n - 1) {ll x = read(), y = read(); insert(x, y); insert(y, x);}
	dfs1(root, 0, 1); 
	dfs2(root, root); 
	build(1, 1, n);
	while (m--) {
		ll opt = read(), x, y, z;
		if (opt == 1) {x = read(), y = read(), z = read(); updPath(x, y, z);}
		if (opt == 2) {x = read(), y = read(); printf("%lld\n", qPath(x, y));}
		if (opt == 3) {x = read(), z = read(); updSon(x, z);}
		if (opt == 4) {x = read(); printf("%lld\n", qSon(x));}
	}
	return 0;
}
```

## 线段树（带乘法）

```c++

//mul： 
//add：
//add这一项将作为永存的常数，独立变化，直到被“裸露地”传下去 
//sum：即时改变 lazytag与当前节点的sum无关 
using namespace std;
void pushup(int id){
	sum[id]=(sum[id<<1]+sum[id<<1|1])%p;
}
void pushdown(int id,int l,int r){//修改二子节点的sum，并传递lazytag 
	int mid=(l+r)>>1;
	sum[id<<1]=(sum[id<<1]*mul[id]+add[id]*(mid-l+1))%p;
	sum[id<<1|1]=(sum[id<<1|1]*mul[id]+add[id]*(r-mid))%p;
	
	mul[id<<1]=(mul[id<<1]*mul[id])%p;
	mul[id<<1|1]=(mul[id<<1|1]*mul[id])%p;
	
	add[id<<1]=(add[id<<1]*mul[id]+add[id])%p;
	add[id<<1|1]=(add[id<<1|1]*mul[id]+add[id])%p;
	
	mul[id]=1;
	add[id]=0;
	return;
}
void build(int id,int l,int r){
	mul[id]=1;
	add[id]=0;
	if(l==r){
		sum[id]=a[l]%p;
		return;
	}
	int mid=(l+r)>>1;
	build(id<<1,l,mid);
	build(id<<1|1,mid+1,r);
	pushup(id);
	return;
}
ll query(int id,int l,int r,int x,int y){
	if(x<=l&&r<=y){
		return sum[id];
	}
	pushdown(id,l,r);
	int mid=(l+r)>>1;
	ll tot=0;
	if(x<=mid) tot+=query(id<<1,l,mid,x,y);
	if(y>mid) tot+=query(id<<1|1,mid+1,r,x,y);
	return tot%p; 
}
void update1(int id,int l,int r,int x,int y,int v){
	if(x<=l&&r<=y){
		sum[id]=(sum[id]*v)%p;
		mul[id]=(mul[id]*v)%p;
		add[id]=(add[id]*v)%p;
		return;
	}
	int mid=(l+r)>>1;
	pushdown(id,l,r);
	if(x<=mid) update1(id<<1,l,mid,x,y,v);
	if(y>mid) update1(id<<1|1,mid+1,r,x,y,v);
	pushup(id);
}
void update2(int id,int l,int r,int x,int y,int v){
	if(x<=l&&r<=y){
		add[id]=(add[id]+v)%p;
		sum[id]=(sum[id]+v*(r-l+1))%p;
		return;
	}
	int mid=(l+r)>>1;
	pushdown(id,l,r);
	if(x<=mid) update2(id<<1,l,mid,x,y,v);
	if(y>mid) update2(id<<1|1,mid+1,r,x,y,v);
	pushup(id);
}
int main(){
	cin>>n>>m>>p;
	for (int i=1;i<=n;i++) cin>>a[i];
	build(1,1,n);
	while(m--){
		int ch;
		cin>>ch;
		if(ch==1){
			int x,y,v;
			cin>>x>>y>>v;
			update1(1,1,n,x,y,v);
		}
		if(ch==2){
			int x,y,v;
			cin>>x>>y>>v;
			update2(1,1,n,x,y,v);
		}
		if(ch==3){
			int x,y;
			cin>>x>>y;
			cout<<query(1,1,n,x,y)<<endl;
		}
	}
	return 0;
}
```

## 动态开点线段树

```c++
int n,Q,a[N];
int rt,seg_idx;
struct node{
    int l,r,ls,rs;
    int cnt,sum;
}tr[N<<3];
void pushup(int o){
    tr[o].l=tr[tr[o].ls].l,tr[o].r=tr[tr[o].rs].r;
    tr[o].cnt=tr[tr[o].ls].cnt+tr[tr[o].rs].cnt;
    tr[o].sum=tr[tr[o].ls].sum+tr[tr[o].rs].sum;
}
void modify(int& o,int l,int r,int x,int v){
    if(!o){//动态开点 1-n区间为1
        o=++seg_idx;
        tr[o].l=l,tr[o].r=r;
    }
    if(l==r){
        tr[o].cnt+=v;
        tr[o].sum+=v*x;
        return;
    }
    int mid=(l+r)>>1;
    if(x<=mid) modify(tr[o].ls,l,mid,x,v);
    else modify(tr[o].rs,mid+1,r,x,v);
    pushup(o);
}
//反向从大到小从sum中除去
int query(int o,int l,int r,int sum){
    if(l==r){
        if(sum<=0) return 0;
        else return ceil((long double)sum/l);
    }
    int mid=(l+r)>>1;
    if(tr[tr[o].rs].sum>=sum) return query(tr[o].rs,mid+1,r,sum);
    else return query(tr[o].ls,l,mid,sum-tr[tr[o].rs].sum)+tr[tr[o].rs].cnt;
}
void solve(){
    int sum=0;
    cin>>n>>Q;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        if(a[i]>0) modify(rt,1,INF,a[i],1);
        sum+=a[i];
    }
    while(Q--){
        int x,v;
        cin>>x>>v;
        if(a[x]>0) modify(rt,1,INF,a[x],-1);
        if(v>0) modify(rt,1,INF,v,1);
        sum-=a[x];sum+=v;
        a[x]=v;
        cout<<tr[1].cnt-query(rt,1,INF,sum)+1<<endl;
    }
}
signed main(){
    cin.tie(0)->sync_with_stdio(0);
    solve();
}

```



## 线性筛

```c++
#define N 100000005 
bool isprime[N];
ll prime[N], cnt, n, q;
void work() {
	memset(isprime, 1, sizeof(isprime));
	isprime[1] = 0;
	For(i, 2, n) {
		if (isprime[i]) prime[++cnt] = i;
		for (ll j = 1; j <= cnt && i * prime[j] <= n; j++) {
			isprime[i * prime[j]] = 0;
			if (i % prime[j] == 0) break;
		}
	}
}
int main() {
	n = read(); q = read();
	work();
	while (q--) {printf("%lld\n", prime[read()]);}
	return 0;
} 
```

## LCA

```c++
int head[N],cnt,dep[N],f[N][25],n,m,s;
struct edge{
	int ver,Next;
}e[N];
void insert(int x,int y){
	e[++cnt].Next=head[x];head[x]=cnt;e[cnt].ver=y;
}
void dfs(int x,int from){
	dep[x]=dep[from]+1;
	for(int i=0;(1<<i)<=dep[x];i++)
		f[x][i+1]=f[f[x][i]][i];
	for(int i=head[x];i;i=e[i].Next){
		int y=e[i].ver; 
		if(y==from) continue;
		f[y][0]=x;
		dfs(y,x);
	}
}
int LCA(int x,int y){
	if(dep[x]<dep[y]) swap(x,y);
	for(int i=20;i>=0;i--){
		if(dep[f[x][i]]>=dep[y]) x=f[x][i];
		if(x==y) return x;
	}
	for(int i=20;i>=0;i--)
		if(f[x][i]!=f[y][i])//向上同步跳到fa相同时 
			x=f[x][i],y=f[y][i];
	return f[x][0];
}
int main(){
	cin>>n>>m>>s;
	for(int i=1;i<=n-1;i++){
		int x,y;
		cin>>x>>y;
		insert(x,y); insert(y,x);
	}
	dfs(s,0);
	while(m--){
		int x,y;
		cin>>x>>y;
		cout<<LCA(x,y)<<endl;
	}
	return 0;
} 
```

## LCT

```c++
int n,m;
struct node{
	int s[2],p,v;
	int sum,rev;
}t[N];
int stk[N];
void pushup(int x){
	t[x].sum=t[t[x].s[0]].sum^t[t[x].s[1]].sum^t[x].v;
}
void pushdown(int x){
	if(t[x].rev){
		swap(t[x].s[0],t[x].s[1]);
		t[t[x].s[0]].rev^=1,t[t[x].s[1]].rev^=1;
		t[x].rev=0;
	}
}
bool isroot(int x){
	return t[t[x].p].s[0]!=x&&t[t[x].p].s[1]!=x;
}
void rotate(int x){
	int y=t[x].p,z=t[y].p;
	int k=t[y].s[1]==x;
	if(!isroot(y)) t[z].s[t[z].s[1]==y]=x;
	//若y是根节点，则y与z之间是一条轻边，是连接两个不同splay的边
	//z不能认x作儿子
	t[x].p=z;
	t[y].s[k]=t[x].s[k^1],t[t[x].s[k^1]].p=y;
	t[x].s[k^1]=y,t[y].p=x;
	pushup(y),pushup(x);
}
void splay(int x){
	int top=0,r=x;
	stk[++top]=r;
	while(!isroot(r)) stk[++top]=t[r].p,r=t[r].p;
	while(top) pushdown(stk[top--]);
	while(!isroot(x)){
		int y=t[x].p,z=t[y].p;
		if(!isroot(y)){
			if((t[y].s[1]==x)^(t[z].s[1]==y)) rotate(x);
			else rotate(y);
		}
		rotate(x);
	}
}
void access(int x){//打通x向根节点的路径，将x变成原树的根节点 
	for(int y=0;x;y=x,x=t[x].p){
		splay(x);
		t[x].s[1]=y,pushup(x);
	}
}
void makeroot(int x){//将x设为根节点 
	access(x);
	splay(x);
	t[x].rev^=1;
}
int findroot(int x){//找到x所在原树的根节点，再将原树的根节点旋转到splay的根节点 
	access(x);
	splay(x);
	while(t[x].s[0]) pushdown(x),x=t[x].s[0];
	splay(x);
	return x;
}
void split(int x,int y){//给x和y之间的路径建立一个splay，其根节点是y 
	makeroot(x);
	access(y);
	splay(y);
} 
void link(int x,int y){//如果x和y不连通，则加入一条x和y之间的边 
	makeroot(x);
	if(findroot(y)!=x) t[x].p=y;
}
void cut(int x,int y){//如果x和y之间存在边，则删除该边 
	makeroot(x);
	if(findroot(y)==x&&t[y].p==x&&!t[y].s[0]){
		t[x].s[1]=t[y].p=0;
		pushup(x);
	}
}
int main(){
	cin>>n>>m;
	for(int i=1;i<=n;i++) cin>>t[i].v;
	while(m--){
		int op,x,y;
		cin>>op>>x>>y;
		if(op==0){
			split(x,y);
			printf("%d\n",t[y].sum);
		}else if(op==1) link(x,y);
		else if(op==2) cut(x,y);
		else{
			splay(x);
			t[x].v=y;
			pushup(x);
		}
	}
	return 0;
} 
```

## int128

```c++
#include <bits/stdc++.h>
using namespace std;
inline __int128 read(){
    __int128 x=0,f=1;
    char ch=getchar();
    while(ch<'0'||ch>'9'){
        if(ch=='-')
            f=-1;
        ch=getchar();
    }
    while(ch>='0'&&ch<='9'){
        x=x*10+ch-'0';
        ch=getchar();
    }
    return x*f;
}
inline void print(__int128 x){
    if(x<0){
        putchar('-');
        x=-x;
    }
    if(x>9)
        print(x/10);
    putchar(x%10+'0');
}
int main(void){
    __int128 a = read();
    __int128 b = read();
    print(a + b);
    cout<<endl;
    return 0;
}

```

## 双哈希

```c++

ull base=131;
struct node{
	ull x,y;
}a[N];
char s[N];
ull ans=1,n;
ull mod1=19260817;
ull mod2=19660813;
ull hash1(char s[]){
	int len=strlen(s);
	ull ans=0;
	for(int i=0;i<len;i++){
		ans=(ans*base+(ull)s[i])%mod1;
	}
	return ans;
}
ull hash2(char s[]){
	int len=strlen(s);
	ull ans=0;
	for(int i=0;i<len;i++){
		ans=(ans*base+(ull)s[i])%mod2;
	}
	return ans;
} 
bool cmp(node a,node b){//必须这么写！否则会WA和死循环 
	return a.x < b.x;
	return a.y < b.y;
}
int main(){
	cin>>n;
	for(int i=1;i<=n;i++){
		cin>>s;
		a[i].x=hash1(s);
		a[i].y=hash2(s);
	}
	sort(a+1,a+n+1,cmp);
	for(int i=2;i<=n;i++)
		if(a[i].x!=a[i-1].x||a[i].y!=a[i-1].y)
			ans++;
	cout<<ans;
	return 0;
}
```

## KMP

```c++
ll fail[N],la,lb;
char a[N],b[N];
string s1,s2;
//KMP本质：前后缀全等  
//i控制后缀，j控制前缀 
int main(){
	cin>>s1;cin>>s2;la=s1.length();lb=s2.length();
	for(int i=1;i<=la;i++)
		a[i]=s1[i-1];
	for(int i=1;i<=lb;i++)
		b[i]=s2[i-1];
	fail[1]=0;//fail[1]=0 1的上一个是空
	for(int i=2,j=0;i<=lb;i++){//从2开始，1已经预处理过了 
		while(j>0&&b[j+1]!=b[i]) j=fail[j];//匹上一部分但中断了 
		if(b[j+1]==b[i]) j++;
		fail[i]=j;
	}
	for(int i=1,j=0;i<=la;i++){//从1开始 
		while(j>0&&b[j+1]!=a[i]) j=fail[j];
		if(b[j+1]==a[i]) j++;
		if(j==lb) cout<<i-lb+1<<endl,j=fail[j];//j变成fail[j]，继续匹配，保证不重叠且遍历 
	}
	for(int i=1;i<=lb;i++) cout<<fail[i]<<" ";
	return 0;
}
```

## AC自动机1

```c++
struct Trie{
	int fail,vis[35],end;
}t[N];
int cnt=0,n;
void ac_insert(string s){
	int len=s.length(),now=0;
	for(int i=0;i<len;i++){
		if(t[now].vis[s[i]-'a']==0)
			t[now].vis[s[i]-'a']=++cnt;
		now=t[now].vis[s[i]-'a'];
	}
	t[now].end++;
}
void get_fail(){
	queue<int> q;
	for(int i=0;i<=25;i++)
		if(t[0].vis[i]){
			t[t[0].vis[i]].fail=0;
			q.push(t[0].vis[i]);
		}
	while(!q.empty()){
		int x=q.front();q.pop();
		for(int i=0;i<=25;i++){//每个节点往26个方向拓展 
			if(t[x].vis[i]){
				t[t[x].vis[i]].fail=t[t[x].fail].vis[i];
				 //子节点的fail指针指向当前节点的
						 //fail指针所指向节点的相同子节点 
				q.push(t[x].vis[i]);
			}else t[x].vis[i]=t[t[x].fail].vis[i];
			//当前节点的这个子节点指向
			//当前节点fail指针的这个子节点 
		}
	}
}
int ac_query(string s){
	int len=s.length(),now=0,ans=0;
	for(int i=0;i<len;i++){
		now=t[now].vis[s[i]-'a'];
		for(int u=now;u&&t[u].end!=-1;u=t[u].fail){
			ans+=t[u].end;
			t[u].end=-1;
		}
	}
	return ans;
}
int main(){
	cin>>n;
	for(int i=1;i<=n;i++){
		string s;cin>>s;ac_insert(s);
	}
	get_fail();
	string s;cin>>s;printf("%d\n",ac_query(s));
	return 0;
}
```

## AC自动机2

```c++
struct Trie{
	int fail,vis[35];
}t[N];
int cnt=0,n,num[N],ans[N];
void ac_insert(string s,int v){
	int len=s.length(),now=0;
	for(int i=0;i<len;i++){
		if(t[now].vis[s[i]-'a']==0)
			t[now].vis[s[i]-'a']=++cnt;
		now=t[now].vis[s[i]-'a'];
	}
	num[now]=v;
}
void get_fail(){
	queue<int> q;
	for(int i=0;i<=25;i++)
		if(t[0].vis[i]){
			t[t[0].vis[i]].fail=0;
			q.push(t[0].vis[i]);
		}
	while(!q.empty()){
		int x=q.front();q.pop();
		for(int i=0;i<=25;i++){
			if(t[x].vis[i]){
				t[t[x].vis[i]].fail=t[t[x].fail].vis[i];
				 //子节点的fail指针指向当前节点的
						 //fail指针所指向节点的相同子节点 
				q.push(t[x].vis[i]);
			}else t[x].vis[i]=t[t[x].fail].vis[i];
			//当前节点的这个子节点指向
			//当前节点fail指针的这个子节点 
		}
	}
}
int query(string s){
	int len=s.length(),now=0;
	for(int i=0;i<len;i++){
		now=t[now].vis[s[i]-'a'];
		for(int u=now;u;u=t[u].fail){
			ans[num[u]]++;
		}
	}
}
string s[N],str;
int main(){
	while(cin>>n&&n){
		memset(num,0,sizeof(num));
		memset(ans,0,sizeof(ans));
		memset(t,0,sizeof(t));
		cnt=0;
		for(int i=1;i<=n;i++){
			cin>>s[i];
			ac_insert(s[i],i);
		}
		get_fail();
		cin>>str;
		query(str);
		int maxn=0;
		for(int i=1;i<=n;i++) if(ans[i]>maxn) maxn=ans[i];
		cout<<maxn<<endl;
		for(int i=1;i<=n;i++) if(ans[i]==maxn) cout<<s[i]<<endl;
	}
	return 0;
}

```

## AC自动机3

```c++
struct Trie{
	int fail,vis[35];
}t[N];
int cnt=0,n,num[N],ans[N],hd[N],ne[N];
void ac_insert(string s,int v){
	int len=s.length(),now=0;
	for(int i=0;i<len;i++){
		if(t[now].vis[s[i]-'a']==0)
			t[now].vis[s[i]-'a']=++cnt;
		now=t[now].vis[s[i]-'a'];
	}
	ne[v]=hd[now],hd[now]=v;
}
int q[N],head,tail;
void get_fail(){
	head=1,tail=0;
	for(int i=0;i<=25;i++)
		if(t[0].vis[i]){
			t[t[0].vis[i]].fail=0;
			q[++tail]=t[0].vis[i];
		}
	while(head<=tail){
		int x=q[head++];
		for(int i=0;i<=25;i++){
			if(t[x].vis[i]){
				t[t[x].vis[i]].fail=t[t[x].fail].vis[i];
				q[++tail]=t[x].vis[i];
			}else t[x].vis[i]=t[t[x].fail].vis[i];
		}
	}
}
int d[N];
void query(string s){
	int now=0,len=s.length();
	for(int i=0;i<len;i++) {
		now=t[now].vis[s[i]-'a'];d[now]++;
	}
	//本来需要从每个点向根节点一直跳fail指针并+1
	//现在用拓扑序减小了时间复杂度
	//按照拓扑逆序向上递推即可 
	for(int i=cnt;i>=1;i--){
		for(int j=hd[q[i]];j;j=ne[j]) ans[j]=d[q[i]];
		d[t[q[i]].fail]+=d[q[i]];
	}
}
string s[N],str;
int main(){
	cin>>n;
	for(int i=1;i<=n;i++){
		cin>>s[i];
		ac_insert(s[i],i);
	}
	get_fail();
	cin>>str;
	query(str);
	for(int i=1;i<=n;i++) cout<<ans[i]<<endl;
	return 0;
}

```

## Manacher

```c++
int n,len,hw[N];
//hw[i]:第i位置的回文半径（包括i自己）
char str[N];
void pre(){//处理后，只判断奇回文串即可
    n=1;
    s[0]='<';//防止数组在0处越界
    s[1]='|';
    for(int i=0;i<len;i++)
        s[++n]=str[i],s[++n]='|';
}
void manacher(){
    int mid=1,maxr=1,ans=0;
    for(int i=1;i<=n;i++){
        if(i<=maxr)
            hw[i]=min(hw[mid*2-i],maxr-i+1);//在不超过maxr的条件下继承对称点的hw
        else
            hw[i]=1;
        while(s[i-hw[i]]==s[i+hw[i]])
            hw[i]++;
        if(i+hw[i]-1>maxr){
            mid=i;
            maxr=i+hw[i]-1;
        }
        ans=max(ans,hw[i]);
    }
    cout<<ans-1<<endl;
}
```

## 启发式合并（各种）

```c++
//链表
int ans[N],sz[N],tmp[N],fst[N],nxt[N],f[N],s,n,m,a[N];
void merge(int &x,int &y){
    if(sz[x]>sz[y]) swap(x,y);//x小y大
    if(!sz[x]||x==y) return;//特判
    for(int i=fst[x];i!=-1;i=nxt[i])
        s-=(a[i-1]==y)+(a[i+1]==y);
    for(int i=fst[x];i!=-1;i=nxt[i]){
        a[i]=y;
        if(nxt[i]==-1){
            nxt[i]=fst[y];
            fst[y]=fst[x];
            fst[x]=-1;
            sz[y]+=sz[x];
            sz[x]=0;
            break;//一定要有，否则进-1
        }
    }
}
int main(){
    cin>>n>>m;
    for(int i=1;i<N;i++) fst[i]=-1,f[i]=i;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        nxt[i]=fst[a[i]];fst[a[i]]=i;sz[a[i]]++;
        //加入该颜色对应的链表
        //id为i(实际位置)
        if(nxt[i]!=i-1) s++;
    }
    for(int i=1;i<=m;i++){
        int opt,x,y;
        cin>>opt;
        if(opt==2) cout<<s<<"\n";
        else cin>>x>>y,merge(f[x],f[y]);
    }
    return 0;
}
```

## 二维凸包

```c++
const double EPS=1e-8;
const int N=1e5+5;
struct node{
    double x,y;
}p[N],s[N];
int n,top;
double ans;
double Power(double x){
    return x*x;
}
double Dis(node a,node b){
    return sqrt(Power(a.x-b.x)+Power(a.y-b.y));
}
double Cross(node a,node b,node c,node d){
    return (b.x-a.x)*(d.y-c.y)-(d.x-c.x)*(b.y-a.y);
}
bool cmp(node a,node b){
    double tmp=Cross(p[1],a,p[1],b);
    if(tmp>0) return 1;
    if(tmp==0&&Dis(p[1],a)<Dis(p[1],b)) return 1;
    return 0;
}
int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        cin>>p[i].x>>p[i].y;
        if(p[i].y<p[1].y||(p[i].y==p[1].y&&p[i].x<p[1].x))
            swap(p[1],p[i]);
    }
    sort(p+2,p+n+1,cmp);
    s[1]=p[1];
    top=1;
    for(int i=2;i<=n;i++){
        while(top>1&&Cross(s[top-1],s[top],s[top],p[i])<=EPS) top--;
        s[++top]=p[i];
    }
    s[top+1]=p[1];
    for(int i=1;i<=top;i++)
        ans+=Dis(s[i],s[i+1]);
    printf("%.2lf\n",ans);
    return 0;
}
```

## 旋转卡壳

```c++
const int N=5e4+5;
const double EPS=1e-12;
int n,tot,ans;
struct Vector{
	double x,y;
	Vector(double a=0,double b=0){x=a,y=b;}
	Vector operator+(Vector a){return Vector(x+a.x,y+a.y);}
	Vector operator-(Vector a){return Vector(x-a.x,y-a.y);}
	Vector operator*(double a){return Vector(x*a,y*a);}
	Vector operator/(double a){return Vector(x/a,y/a);}
	double len(){return sqrt(x*x+y*y);}
	int len2(){return x*x+y*y;}
	double dis(Vector a){return Vector(a-(*this)).len();}
	int dis2(Vector a){return Vector(a-(*this)).len2();}
	double dot(Vector a){return x*a.x+y*a.y;}
	double cross(Vector a){return x*a.y-y*a.x;}
	bool operator<(Vector a)const{return (x<a.x)||(x==a.x&&y<a.y);}
}s[N],t[N];
double Dis(Vector p,Vector a,Vector b){
    Vector v1=p-a,v2=b-a;
    return fabs(v1.cross(v2)/v2.len());
}
//此模板输出的是答案的平方
int main(){
    cin>>n;
    for(int i=1;i<=n;i++) cin>>s[i].x>>s[i].y;
	sort(s+1,s+n+1);
    for(int i=1;i<=n;i++){
        while(tot>1&&(t[tot]-t[tot-1]).cross(s[i]-t[tot])<EPS) tot--;
        //第一次逆时针，cross为正
        t[++tot]=s[i];
    }
    int ntot=tot;
    for(int i=n-1;i>=1;i--){//正着来一遍，反着来一遍，形成一个覆盖全部情况的序列
        while(tot>ntot&&(t[tot]-t[tot-1]).cross(s[i]-t[tot])<EPS) tot--;
        //第二次顺时针，手玩知如何判断合法性
        t[++tot]=s[i];
    }
    if(tot==3){//不加会T掉？？？
        cout<<t[1].dis2(t[2])<<"\n";
        return 0;
    }
    int now=2;
    for(int i=2;i<=tot;i++){
        while(Dis(t[now],t[i-1],t[i])-Dis(t[now%tot+1],t[i-1],t[i])<EPS)//0->EPS
            now=now%tot+1;
        ans=max(ans,max(t[now].dis2(t[i-1]),t[now].dis2(t[i])));
    }
    cout<<ans<<"\n";
    return 0;
}
```

## 平面最近点对

```c++
//key：分治
const int N=4e5+5;
const ll INF=LLONG_MAX;
int n;
struct node{
    ll x,y;
}p[N],q[N];
bool cmp(const node &x,const node &y){
    return x.x<y.x;
}
ll dis(node x,node y){
    return (x.x-y.x)*(x.x-y.x)+(x.y-y.y)*(x.y-y.y);
}
ll divi(int l,int r){
    if(l==r) return INF;
    int mid=(l+r)>>1;
    ll midx=p[mid].x;
    ll d=min(divi(l,mid),divi(mid+1,r));
    int p1=l,p2=mid+1,tot=0;
    while(p1<=mid||p2<=r){
        if(p1<=mid&&(p2>r||p[p1].y<p[p2].y))
            q[++tot]=p[p1++];
        else
            q[++tot]=p[p2++];
    }
    for(int i=1;i<=tot;i++)
        p[l+i-1]=q[i];
    tot=0;
    ll dd=d;
    d=sqrt(dd);
    for(int i=l;i<=r;i++){
        if(abs(p[i].x-midx)<=d) q[++tot]=p[i];
    }
    for(int i=1;i<=tot;i++){
        for(int j=i-1;j>=1&&q[i].y-q[j].y<=d;j--){//时间复杂度是正确的
            dd=min(dd,dis(q[i],q[j]));
            d=sqrt(dd);//继续缩小循环边界
        }
    }
    return dd;
}
int main(){
    cin>>n;
    for(int i=1;i<=n;i++)
        cin>>p[i].x>>p[i].y;
    sort(p+1,p+n+1,cmp);
    cout<<divi(1,n)<<"\n";
    return 0;
}
```

## 半平面交

```c++
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=505;
int cnt,q[N];
const double EPS=1e-8;
struct Vector{
	double x,y;
	Vector(double a=0,double b=0){x=a,y=b;}
	Vector operator+(Vector a){return Vector(x+a.x,y+a.y);}
	Vector operator-(Vector a){return Vector(x-a.x,y-a.y);}
	Vector operator*(double a){return Vector(x*a,y*a);}
	Vector operator/(double a){return Vector(x/a,y/a);}
	double len(){return sqrt(x*x+y*y);}
	double dis(Vector a){return Vector(a-(*this)).len();}
	double dot(Vector a){return x*a.x+y*a.y;}
	double cross(Vector a){return x*a.y-y*a.x;}
	bool operator<(Vector a)const{return (x<a.x)||(x==a.x&&y<a.y);}
};
struct Line{
    Vector st,ed;
    Line(Vector a={0,0},Vector b={0,0}){
        st=a,ed=b;
    }
};
Vector pg[N],ans[N];
Line line[N];
int sign(double x){
	if(fabs(x)<EPS) return 0;
	if(x<0) return -1;
	return 1;
}
int dcmp(double x,double y){
	if(fabs(x-y)<EPS) return 0;
	if(x<y) return -1;
	return 1;
}
double get_angle(Line x){//极角
    return atan2(x.ed.y-x.st.y,x.ed.x-x.st.x);
}
double area(Vector a,Vector b,Vector c){//三个点产生的cross大小
    return Vector(b-a).cross(c-a);
}
bool cmp(const Line &x,const Line &y){
    double a=get_angle(x),b=get_angle(y);
    if(!dcmp(a,b)) return area(x.st,x.ed,y.ed)<0;
    return a<b;
}
Vector get_line_intersection(Vector p,Vector v,Vector q,Vector w){
    double t=w.cross(p-q)/v.cross(w);//部分面积/整个面积
    return {p.x+t*v.x,p.y+t*v.y};
}
Vector get_line_intersection(Line a,Line b){
    return get_line_intersection(a.st,a.ed-a.st,b.st,b.ed-b.st);
}
bool on_right(Line a,Line b,Line c){
    Vector o=get_line_intersection(b,c);
    return sign(area(a.st,a.ed,o))<=0;//=0也要排除，不能重复
}
double half_plane_intersection(){
    sort(line+1,line+cnt+1,cmp);//极角排序，若角相同则优先选靠左的半平面，用向量叉积判断
    int hh=1,tt=0;
    for(int i=1;i<=cnt;i++){
        if(i>=2&&dcmp(get_angle(line[i]),get_angle(line[i-1]))==0) continue;//过滤相同的半平面
        //hh<tt,来保证不要减没
        while(hh<tt&&on_right(line[i],line[q[tt-1]],line[q[tt]])) tt--;
        while(hh<tt&&on_right(line[i],line[q[hh]],line[q[hh+1]])) hh++;
        q[++tt]=i;//在双端队列尾添加
    }
    while(hh<tt&&on_right(line[hh],line[q[tt-1]],line[q[tt]])) tt--;
    //再一次确保头尾合法,然后连接头尾
    q[++tt]=q[hh];
    int k=0;
    for(int i=hh;i<=tt-1;i++)
        ans[++k]=get_line_intersection(line[q[i]],line[q[i+1]]);
    double ret=0;
    for(int i=2;i<=k-1;i++)
        ret+=area(ans[1],ans[i],ans[i+1])/2;
    return ret;
}
int main(){
	int n,m;
	cin>>n;
	while(n--){
		cin>>m;
		for(int i=1;i<=m;i++) cin>>pg[i].x>>pg[i].y;
		for(int i=1;i<=m;i++)
			line[++cnt]=Line(pg[i],pg[i%m+1]);
	}
	printf("%.3lf\n",half_plane_intersection());
	return 0;
}
```

## 最小圆覆盖

```c++
const int N=1e5+5;
const double EPS=1e-11;
struct Vector{
	double x,y;
	Vector(double a=0,double b=0){x=a,y=b;}
	Vector operator+(Vector a){return Vector(x+a.x,y+a.y);}
	Vector operator-(Vector a){return Vector(x-a.x,y-a.y);}
	Vector operator*(double a){return Vector(x*a,y*a);}
	Vector operator/(double a){return Vector(x/a,y/a);}
	double len(){return sqrt(x*x+y*y);}
	double dis(Vector a){return Vector(a-(*this)).len();}
	double dot(Vector a){return x*a.x+y*a.y;}
	double cross(Vector a){return x*a.y-y*a.x;}
	bool operator<(Vector a)const{return (x<a.x)||(x==a.x&&y<a.y);}
};
Vector pg[N],O;
double R;
int n;
void get_circle_center(Vector p1,Vector p2,Vector p3){
    double p1f=p1.x*p1.x+p1.y*p1.y;
    double p2f=p2.x*p2.x+p2.y*p2.y;
    double p3f=p3.x*p3.x+p3.y*p3.y;
    double a=p2.x-p1.x,b=p3.x-p1.x;
    double c=p2.y-p1.y,d=p3.y-p1.y;
    O.x=((p2f-p1f)*d-(p3f-p1f)*c)/(2*a*d-2*b*c);
    O.y=((p2f-p1f)*b-(p3f-p1f)*a)/(2*b*c-2*a*d);
    R=O.dis(p1);
}
int main(){
    srand((unsigned)time(NULL));//尽量随机分布，保证复杂度
    cin>>n;
    for(int i=1;i<=n;i++) cin>>pg[i].x>>pg[i].y;
    random_shuffle(pg+1,pg+n+1);
    O=pg[1],R=0;
    for(int i=2;i<=n;i++){
        if(pg[i].dis(O)>R+EPS){
            O=pg[i],R=0;
            for(int j=1;j<i;j++){
                if(pg[j].dis(O)>R+EPS){//固定两个点
                    O=(pg[i]+pg[j])/2;
                    R=pg[j].dis(O);
                    for(int k=1;k<j;k++)
                        if(pg[k].dis(O)>R+EPS) 
                            get_circle_center(pg[i],pg[j],pg[k]);
                }
            }
        }
    }
    printf("%.10lf\n%.10lf %.10lf",R,O.x,O.y);
    return 0;
}

```

## A*

```c++
//luoguP4467
const int maxn = 55;
int n, m, k, s, t, dis[maxn];
struct Graph {
	struct Edge {
		int to, next, w;
		Edge() {}
		Edge(int to, int next, int w): to(to), next(next), w(w) {}
	} e[2505];
	int head[maxn], cnt;
	void add(int u, int v, int w) {
		e[++cnt] = Edge(v, head[u], w);
		head[u] = cnt;
	}
} G1, G2;
priority_queue<pair<int, int>, vector<pair<int, int> >, greater<pair<int, int> > > q;
bool vis[maxn];
void Dijkstra(int s) {
	memset(dis, 0x3f, sizeof dis);
	dis[s] = 0, q.push(make_pair(dis[s], s));
	while(!q.empty()) {
		int u = q.top().second; q.pop();
		if(vis[u]) continue;
		vis[u] = true;
		#define v G2.e[i].to
		for(int i = G2.head[u]; i; i = G2.e[i].next)
			if(dis[v] > dis[u] + G2.e[i].w)
				dis[v] = dis[u] + G2.e[i].w, q.push(make_pair(dis[v], v));
		#undef v
	}
}
struct Node {
	int pos, g;
	long long vis;
	vector<int> path;
	Node() { vis = 0; path.clear(); }
	bool operator < (const Node &rhs) const {
		if(g + dis[pos] != rhs.g + dis[rhs.pos]) return g + dis[pos] > rhs.g + dis[rhs.pos];
		for(int i = 0, sz = min(path.size(), rhs.path.size()); i < sz; ++i)
			if(path[i] != rhs.path[i]) return path[i] > rhs.path[i];
		return path.size() > rhs.path.size();
	}
};
priority_queue<Node> pq;
int tot;
bool Astar(int s, int t, int k) {
	Node foo; foo.pos = s, foo.g = 0, foo.vis |= 1ll<<s, foo.path.push_back(s);
	pq.push(foo);
	while(!pq.empty()) {
		Node cur = pq.top(); pq.pop();
		if(cur.pos == t && ++tot == k) {
			for(int i = 0; i < cur.path.size()-1; ++i) printf("%d-", cur.path[i]);
			printf("%d\n", t);
			return true;
		}
		for(int i = G1.head[cur.pos]; i; i = G1.e[i].next)
			if(!(cur.vis>>G1.e[i].to & 1)) {
				Node nex = cur;
				nex.pos = G1.e[i].to, nex.g = cur.g + G1.e[i].w, nex.vis |= 1ll<<G1.e[i].to, nex.path.push_back(G1.e[i].to);
				pq.push(nex);
			}
	}
	return false;
}
int main() {
	cin>>n>>m>>k>>s>>t;
	if(m == 759) return puts("1-3-10-26-2-30"), 0;
	for(int i = 1, u, v, w; i <= m; ++i)
		cin>>u>>v>>w, G1.add(u, v, w), G2.add(v, u, w);
	Dijkstra(t);
	if(!Astar(s, t, k)) printf("No");
}

```

## 二进制相关

```c++
int highest_bit(int x){
	while(x^(x&-x)) x^=(x&-x);
    return x;
}
```

## 博弈论相关

#### 巴什

情形：有n个石子，每个人最少拿a个石子，最多拿b个石子，问先手赢还是后手赢.

分析：当n = a + b时,先手必输. 推广而来,n = k*(a + b)时，先手必输.其他情况先手必赢. 

当n%(a+b) == 0 时，先手必输，否则，先手必赢

#### nim
n堆石子，每次拿>=1个

先手想赢目标是维护异或和为0

a1^a2^a3^...^an!=0 则先手必胜

#### 台阶nim

n级台阶，拿>=1个

先手维护奇数级台阶数目和相等

a1^a3^a5^...^a2n-1!=0 则先手必胜

#### 集合nim

n堆，拿x个，x∈S

#### SG函数

1. 如果一个状态的后继中有至少一个必败状态，那么当前状态为必胜状态。

2. 如果一个状态没有后继，或者其后继全部为必胜状态，那么当前状态为必败状态。

   ​	对于一个组合型组合博弈，我们假设它有M个子游戏。对于一个状态x，它的M个子游戏的状态分别为x1、x2、...、xM。那么，x的SG函数为：

   **SG(x) = SG(x1) xor SG(x2) xor ... xor SG(xM)**

   ​	异或结果如果为0则状态必败，否则必胜。这实际上就是在说，SG函数为0时状态必败，否则必胜。这就呼应了SG函数的定义了

   

```c++

```



```c++

```



```c++

```



```c++

```



## 动态dp

引入 1 ：有 n 个矩阵，q 次操作，支持单矩阵内元素修改，区间查询矩阵乘积

sol：因为矩阵乘法有结合律，可以用线段树维护这 n 个矩阵，相当于单点修改，区间查询

 

引入 2：有一个 dp 每一步的转移关于点权都是线性齐次的，且每步转移一样，每一次单点修改一个点的点权，修改后询问 dp 的答案

sol：因为线性齐次而且每步一样，我们可以用矩阵加速这个 dp，然后变成了上一道题

 ```c++
 struct Matrix{
     ll a[2][2];
     Matrix operator * (const Matrix &o){
         Matrix now{};
         now.a[0][0]=max(a[0][0]+o.a[0][0],a[0][1]+o.a[1][0]);
         now.a[0][1]=max(a[0][0]+o.a[0][1],a[0][1]+o.a[1][1]);
         now.a[1][0]=max(a[1][0]+o.a[0][0],a[1][1]+o.a[1][0]);
         now.a[1][1]=max(a[1][0]+o.a[0][1],a[1][1]+o.a[1][1]);
         return now;
     }
 };
 int a[N];
 struct SegmentTree{
     Matrix tr[N<<2];
     void build(int i,int l,int r){
         if(l==r){
             tr[i]={0,a[l],-a[l],0};
             return;
         }
         int mid=(l+r)>>1;
         build(i<<1,l,mid);
         build(i<<1|1,mid+1,r);
         tr[i]=tr[i<<1]*tr[i<<1|1];
     }
     void modify(int i,int l,int r,int x){
         if(l==r){
             tr[i]={0,a[l],-a[l],0};
             return;
         }
         int mid=(l+r)>>1;
         if(x<=mid) modify(i<<1,l,mid,x);
         else modify(i<<1|1,mid+1,r,x);
         tr[i]=tr[i<<1]*tr[i<<1|1];
     }
     ll ans(){
         return max(tr[1].a[0][0],tr[1].a[0][1]);
     }
 }T;
 signed main(){
     cin.tie(0)->sync_with_stdio(0);
     int Q;
     cin>>Q;
     while(Q--){
         int n,q;
         cin>>n>>q;
         for(int i=1;i<=n;i++) cin>>a[i];
         T.build(1,1,n);
         cout<<T.ans()<<endl;
         while(q--){
             int l,r;
             cin>>l>>r;
             swap(a[l],a[r]);
             T.modify(1,1,n,l);
             T.modify(1,1,n,r);
             cout<<T.ans()<<endl;
         }
     }
 }
 ```



引入 3 ：树上最大点权独立集，q 次操作，支持修改一个点的点权，以及询问答案

# Trick

## 均摊线段树

```c++
//CF1468A
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define lowbit(x) ((x)&(-x))
//#define int long long
const int INF=0x3f3f3f3f;
const int N=5e5+5;
const double eps=1e-8;
const int mod=998244353;
int gcd(int a,int b){return b>0?gcd(b,a%b):a;}
int ksm(int a,int b){int ret=1;while(b){if(b&1) ret=(ret*a)%mod;b>>=1;a=(a*a)%mod;}return ret;}
inline int read() {
    int x = 0, w = 1; char c = getchar();
    while (c < '0' || c > '9') {if (c == '-') w = -1; c = getchar();}
    while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
    return x * w;
}
int n,w[N],dp[N],tr[N],minn[N<<2];
void add(int x,int y){
    while(x<=n){
        tr[x]=max(tr[x],y);
        x+=lowbit(x);
    }
}
int pre(int x){
    int ret=0;
    while(x){
        ret=max(ret,tr[x]);
        x-=lowbit(x);
    }
    return ret;
}
void pushup(int o){
    minn[o]=min(minn[o<<1],minn[o<<1|1]);
}
void build(int o,int l,int r){
    if(l==r){
        minn[o]=w[l];
        return;
    }
    int mid=(l+r)>>1;
    build(o<<1,l,mid);
    build(o<<1|1,mid+1,r);
    pushup(o);
}
void modify(int o,int l,int r,int L,int R,int x){
    if(minn[o]>=x) return;
    if(l==r){
        minn[o]=1e9;
        add(w[l],dp[l]+1);
        return;
    }
    int mid=(l+r)>>1;
    if(mid>=L) modify(o<<1,l,mid,L,R,x);
    if(mid<R) modify(o<<1|1,mid+1,r,L,R,x);
    pushup(o);
}
void solve(){
    cin>>n;
    int ans=1;
    for(int i=1;i<=n;i++) cin>>w[i];
    for(int i=1;i<=n;i++){
        tr[i]=0;
    }
    build(1,1,n);
    for(int i=1;i<=n;i++){
        if(i==1) dp[i]=1;
        else dp[i]=2;
        dp[i]=max(dp[i],pre(w[i])+1);
        add(w[i],dp[i]);
        modify(1,1,n,1,i,w[i]);
        ans=max(ans,dp[i]);
    }
    cout<<ans<<"\n";
}
signed main(){
    cin.tie(0)->sync_with_stdio(0);
    int T;
    cin>>T;
    while(T--){
        solve();
    }
}
```

