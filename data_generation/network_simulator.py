import requests
import random
import time
import socket
import string
from scapy.all import ARP, Ether, srp, IP, ICMP, UDP, send

# Get the container's hostname and IP
HOSTNAME = socket.gethostname()
MY_IP = socket.gethostbyname(HOSTNAME)

# Network settings
NETWORK_PREFIX = "172.25.0."

# Lists for different network operations
websites = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://x.com",
    "https://www.microsoft.com",
    "https://www.facebook.com",
    "https://www.instagram.com",
    "https://www.reddit.com",
    "https://www.wikipedia.org",
    "https://www.whatsapp.com",
    "https://www.amazon.com",
    "https://www.linkedin.com",
    "https://www.netflix.com",
    "https://www.yahoo.com",
    "https://www.bing.com",
    "https://www.office.com",
    "https://www.microsoftonline.com",
    "https://www.live.com",
    "https://www.pinterest.com",
    "https://www.quora.com",
    "https://www.weather.com",
    "https://www.apple.com",
    "https://www.nytimes.com",
    "https://www.zoom.us",
    "https://www.duckduckgo.com",
    "https://www.chatgpt.com",
    "https://www.canva.com",
    "https://www.fandom.com",
    "https://www.samsung.com",
    "https://www.discord.com",
    "https://www.roblox.com",
    "https://www.msn.com",
    "https://www.cnn.com",
    "https://www.ebay.com",
    "https://www.booking.com",
    "https://www.paypal.com",
    "https://www.stackoverflow.com",
    "https://www.medium.com",
    "https://www.imdb.com",
    "https://www.slack.com"
]

dns_queries = [
    "google.com",
    "yahoo.com",
    "bing.com",
    "cloudflare.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "amazon.com",
    "wikipedia.org",
    "twitter.com",
    "linkedin.com",
    "netflix.com",
    "reddit.com",
    "pinterest.com",
    "microsoft.com",
    "apple.com",
    "whatsapp.com",
    "zoom.us",
    "paypal.com",
    "ebay.com",
    "live.com",
    "office.com",
    "duckduckgo.com",
    "booking.com",
    "espn.com",
    "chatgpt.com",
    "canva.com",
    "fandom.com",
    "samsung.com",
    "discord.com",
    "roblox.com",
    "msn.com",
    "cnn.com",
    "bbc.com",
    "nytimes.com",
    "stackoverflow.com",
    "medium.com",
    "imdb.com",
    "slack.com"
]

# Function to generate a random string
def random_message(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Function to perform an ARP scan
def send_arp_request():
    # Generate 4 random /24 subnets
    subnets = []
    for _ in range(4):
        subnet = f"10.{random.randint(0, 254)}.{random.randint(0, 254)}.0/24"
        subnets.append(subnet)

    # Generate 1 unique random IP addresses from the selected subnets
    target_ips = set()
    while len(target_ips) < 1:
        subnet = random.choice(subnets)
        base_ip = subnet.split('/')[0]
        octets = base_ip.split('.')
        # Generate a random host IP in the subnet (excluding network and broadcast addresses)
        host_ip = f"{octets[0]}.{octets[1]}.{octets[2]}.{random.randint(1, 254)}"
        target_ips.add(host_ip)

    # Prepare ARP requests
    arp_requests = []
    for ip in target_ips:
        arp = ARP(pdst=ip)
        ether = Ether(dst="ff:ff:ff:ff:ff:ff")
        packet = ether / arp
        arp_requests.append(packet)

    # Send ARP requests and collect responses
    answered, _ = srp(arp_requests, timeout=2, verbose=False)
    devices = [f"MAC: {resp.hwsrc}, IP: {resp.psrc}" for _, resp in answered]
    print("[ARP] Devices found:", devices)    

# Function to perform a single HTTP request
def send_http_request():
    url = random.choice(websites)
    try:
        response = requests.get(url, timeout=1)
        print(f"[HTTP] {url} -> {response.status_code}")
    except requests.RequestException as e:
        print(f"[HTTP] Error: {e}")

# Function to perform a single DNS query
def send_dns_query():
    domain = random.choice(dns_queries)
    try:
        ip = socket.gethostbyname(domain)
        print(f"[DNS] {domain} -> {ip}")
    except socket.gaierror:
        print(f"[DNS] Failed to resolve {domain}")

# Function to get a random IP in the Docker network
def get_random_host():
    while True:
        random_ip = f"{NETWORK_PREFIX}{random.randint(2, 33)}"
        if random_ip != MY_IP:
            return random_ip

# Function to send an ICMP echo request
def send_icmp_packet():
    dst_ip = get_random_host()
    packet = IP(dst=dst_ip) / ICMP()
    send(packet, verbose=False)
    print(f"[ICMP] Sent ICMP request to {dst_ip}")

# Function to send a UDP packet with a random payload
def send_udp_packet():
    for _ in range(1):
        dst_ip = get_random_host()
        src_port = random.randint(1024, 65535)
        dst_port = random.randint(1024, 65535)
        payload = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=random.randint(12,100)))
        packet = IP(dst=dst_ip) / UDP(sport=src_port, dport=dst_port) / payload
        send(packet, verbose=False)
        print(f"[UDP] Sent packet to {dst_ip}:{dst_port} from port {src_port} with payload: {payload}")

# Function to randomly execute one network operation every 2-5 seconds
def simulate_network_traffic():
    operations = {
        "http": send_http_request,
        "dns": send_dns_query,
        "arp": send_arp_request,
        "dns2": send_dns_query,      # added this to have more dns queries
        "icmp": send_icmp_packet,
        "http2": send_http_request,  # added this to have more http requests
        "udp": send_udp_packet
    }
    while True:
        action = random.choice(list(operations.keys()))
        operations[action]()
        time.sleep(random.uniform(8, 12))  # 10 to 15 -> 10k packets per nearly 300 sec when kept 20 hosts -> 

# Start network traffic simulation
simulate_network_traffic()
